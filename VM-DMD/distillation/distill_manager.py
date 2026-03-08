from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .teacher_wrapper import build_teacher


class DirectionWeightNet(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_dirs: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_dirs),
            nn.Softmax(dim=-1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C, H, W]
        Returns:
            w: [num_dirs]
        """
        x = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B, C]
        w = self.net(x)                               # [B, num_dirs]
        return w.mean(dim=0)                          # [num_dirs]


class RelationalDistillationManager:
    def __init__(self, cfg: Dict):
        self.enabled = cfg.get('enable', False)
        if not self.enabled:
            self.teacher = None
            return
        teacher_cfg = cfg.get('teacher', {})
        self.teacher = build_teacher(teacher_cfg).cuda()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

        self.distill_weight = cfg.get('loss_weight', 1.0)
        self.loss_type = cfg.get('loss_type', 'cosine')
        self._raw_stage_weights: List[float] = cfg.get('stage_weights', [1.0, 1.0, 1.0, 1.0])
        self._stage_weight_warning_issued = False
        inferred_stage_count = self._infer_teacher_stage_count()
        if inferred_stage_count <= 0:
            inferred_stage_count = len(self._raw_stage_weights) or 4
        self.stage_weights: List[float] = self._normalize_stage_weights(
            self._raw_stage_weights,
            inferred_stage_count
        )
        self.teacher_stage_count = inferred_stage_count
        self.log_dir_weights = cfg.get('log_dir_weights', False)

        self.dir_weight_nets: List[DirectionWeightNet] = []

        self._dir_weight_sums: List[torch.Tensor] = []
        self._dir_weight_counts: List[int] = []
        self._dir_weight_last: List[torch.Tensor] = []
        self._init_dir_weight_buffers(self.teacher_stage_count)

        self.dir_weight_net: DirectionWeightNet = None

    @torch.no_grad()
    def _extract_teacher_features(self, images: torch.Tensor):
        teacher_feats = self.teacher(images)
        return [feat for feat in teacher_feats]

    def _feature_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor):
        if self.loss_type == 'mse':
            return F.mse_loss(student_feat, teacher_feat)

        s = student_feat.view(student_feat.size(0), student_feat.size(1), -1).transpose(1, 2).contiguous()
        t = teacher_feat.view(teacher_feat.size(0), teacher_feat.size(1), -1).transpose(1, 2).contiguous()
        s = F.normalize(s, dim=2)
        t = F.normalize(t, dim=2)
        cos_sim = F.cosine_similarity(s, t, dim=2)
        return 1 - cos_sim.mean()

    def compute_loss(
        self,
        images: torch.Tensor,
        student_feats: List[torch.Tensor],
        student_model,
        dir_feats: List[List[torch.Tensor]] = None,
    ):
        if not self.enabled:
            return images.new_tensor(0.0)
        if self.teacher is None:
            return images.new_tensor(0.0)

        with torch.no_grad():
            teacher_feats = self._extract_teacher_features(images)

        num_stages = len(teacher_feats)
        if num_stages != len(self.stage_weights):
            self.stage_weights = self._normalize_stage_weights(self._raw_stage_weights, num_stages)
            self.teacher_stage_count = num_stages
            self._init_dir_weight_buffers(num_stages)

        if dir_feats is None:
            del teacher_feats
            return images.new_tensor(0.0)

        total_loss = images.new_tensor(0.0)
        total_weight = 0.0

        for stage_idx in range(min(num_stages, len(dir_feats))):
            s_dirs = dir_feats[stage_idx]  # List[Tensor], len=4
            stage_weight = self.stage_weights[stage_idx] if stage_idx < len(self.stage_weights) else 1.0
            if not s_dirs or stage_weight <= 0:
                continue

            t = teacher_feats[stage_idx]   # [B, C_t, H_t, W_t]

            t_hw = t
            t_wh = t.permute(0, 1, 3, 2).contiguous()
            t_hrev = t.flip(-1)
            t_wrev = t.flip(-2)
            teacher_dirs = [t_hw, t_wh, t_hrev, t_wrev]

            dir_weight_net = self._get_dir_weight_net(stage_idx, t.size(1), t.device)
            dir_weights = dir_weight_net(t).detach()  # [4]
            dir_weights_cpu = dir_weights.detach().cpu()
            self._ensure_dir_weight_capacity(stage_idx + 1)
            self._dir_weight_sums[stage_idx] += dir_weights_cpu
            self._dir_weight_counts[stage_idx] += 1
            self._dir_weight_last[stage_idx] = dir_weights_cpu
            if self.log_dir_weights:
                weights_str = ", ".join(f"{w:.4f}" for w in dir_weights_cpu.tolist())
                print(f"[RelKD] dir_weights stage {stage_idx}: [{weights_str}]")

            if hasattr(student_model, "stage_adapters") and student_model.stage_adapters is not None:
                if stage_idx < len(student_model.stage_adapters):
                    adapter = student_model.stage_adapters[stage_idx]
                    s_dirs_adapted = [adapter(f) for f in s_dirs]
                else:
                    s_dirs_adapted = s_dirs
            else:
                s_dirs_adapted = s_dirs

            for dir_idx, s_feat_dir in enumerate(s_dirs_adapted):
                if dir_idx >= len(teacher_dirs):
                    break
                t_dir = teacher_dirs[dir_idx]

                s_resized = s_feat_dir
                if s_resized.shape[-2:] != t_dir.shape[-2:]:
                    s_resized = F.interpolate(
                        s_resized,
                        size=t_dir.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                loss_dir = self._feature_loss(s_resized, t_dir)
                w = stage_weight * dir_weights[dir_idx]
                total_loss = total_loss + w * loss_dir
                total_weight += float(w)

        del teacher_feats

        if total_weight == 0:
            return images.new_tensor(0.0)
        return self.distill_weight * (total_loss / total_weight)

    def pop_dir_weight_stats(self) -> List[Tuple[int, List[float]]]:

        if not self.enabled or not self._dir_weight_sums:
            return []
        stats: List[Tuple[int, List[float]]] = []
        for idx, (sum_tensor, count) in enumerate(zip(self._dir_weight_sums, self._dir_weight_counts)):
            if count > 0:
                avg = (sum_tensor / count).tolist()
                stats.append((idx, avg))
            else:
                stats.append((idx, []))
            sum_tensor.zero_()
            self._dir_weight_counts[idx] = 0
        return stats

    def _init_dir_weight_buffers(self, stage_count: int):
        self._dir_weight_sums = [torch.zeros(4) for _ in range(stage_count)]
        self._dir_weight_counts = [0 for _ in range(stage_count)]
        self._dir_weight_last = [torch.zeros(4) for _ in range(stage_count)]

    def _ensure_dir_weight_capacity(self, target_len: int):
        current = len(self._dir_weight_sums)
        if target_len <= current:
            return
        for _ in range(target_len - current):
            self._dir_weight_sums.append(torch.zeros(4))
            self._dir_weight_counts.append(0)
            self._dir_weight_last.append(torch.zeros(4))

    def _get_dir_weight_net(self, stage_idx: int, in_dim: int, device: torch.device) -> DirectionWeightNet:
        self._ensure_dir_weight_list_capacity(stage_idx + 1)
        net = self.dir_weight_nets[stage_idx]
        if net is None or net.net[0].in_features != in_dim:
            net = DirectionWeightNet(in_dim=in_dim, num_dirs=4).to(device)
            self.dir_weight_nets[stage_idx] = net
        return net

    def _ensure_dir_weight_list_capacity(self, target_len: int):
        current = len(self.dir_weight_nets)
        if target_len <= current:
            return
        self.dir_weight_nets.extend([None] * (target_len - current))

    def _infer_teacher_stage_count(self) -> int:

        if self.teacher is None:
            return 0
        backbone = getattr(self.teacher, 'backbone', None)
        if backbone is not None:
            out_indices = getattr(backbone, 'out_indices', None)
            if isinstance(out_indices, (tuple, list)) and len(out_indices) > 0:
                return len(out_indices)
            layers = getattr(backbone, 'layers', None)
            if isinstance(layers, (tuple, list)) and len(layers) > 0:
                return len(layers)
        return 0

    def _normalize_stage_weights(self, weights: List[float], target_len: int) -> List[float]:

        if target_len <= 0:
            return list(weights)
        weights = list(weights or [])
        if not weights:
            weights = [1.0]

        if len(weights) == target_len:
            return weights

        if len(weights) > target_len:
            if not self._stage_weight_warning_issued:
                print(f"[RelKD] stage_weights 长度 {len(weights)} 大于教师 stage 数 {target_len}，将自动截断。")
                self._stage_weight_warning_issued = True
            return weights[:target_len]

        # len(weights) < target_len
        extend_value = weights[-1] if weights else 1.0
        extended = weights + [extend_value] * (target_len - len(weights))
        if not self._stage_weight_warning_issued:
            print(f"[RelKD] The length of stage_weights {len(weights)} is less than the number of teacher stages {target_len},"
                  f"Will automatically complete using the last weight {extend_value}.")
            self._stage_weight_warning_issued = True
        return extended

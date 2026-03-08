import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHeadHook:

    def __init__(self, stage_idx: int, block_idx: int, head_idx: int):
        self.stage_idx = stage_idx
        self.block_idx = block_idx
        self.head_idx = head_idx
        self.attention_map = None
        
    def __call__(self, module, input, output):

        if hasattr(module, 'attention_weights'):
            attn_weights = module.attention_weights  # [B_, num_heads, N, N]
            if attn_weights.dim() == 4 and attn_weights.shape[1] > self.head_idx:

                head_attn = attn_weights[:, self.head_idx, :, :]  # [B_, N, N]
                self.attention_map = head_attn.detach()


def modify_window_attention_to_save_attn():

    try:
        from mmseg.models.backbones.swin_transformer import WindowAttention
        

        if hasattr(WindowAttention.forward, '_modified'):
            return
            
        original_forward = WindowAttention.forward
        
        def forward_with_hook(self, x, mask=None):
            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], 
                self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            # attn shape: [B_, num_heads, N, N]
            self.attention_weights = attn.detach()

            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        
        forward_with_hook._modified = True
        WindowAttention.forward = forward_with_hook
    except Exception as e:

        pass 


def convert_attention_to_spatial_map(attn_weights: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:

    if attn_weights.dim() == 2:

        B_ = 1
        N = attn_weights.shape[0]
        attn_weights = attn_weights.unsqueeze(0)
    else:
        B_, N, _ = attn_weights.shape

    avg_attn = attn_weights.mean(dim=1)  
    
    window_attn = avg_attn.view(B_, window_size, window_size)

    num_windows_h = (H + window_size - 1) // window_size
    num_windows_w = (W + window_size - 1) // window_size
    

    full_attn_map = torch.zeros(B_, num_windows_h * window_size, num_windows_w * window_size,
                                device=attn_weights.device, dtype=attn_weights.dtype)
    
    for h in range(num_windows_h):
        for w in range(num_windows_w):
            h_start = h * window_size
            h_end = min(h_start + window_size, num_windows_h * window_size)
            w_start = w * window_size
            w_end = min(w_start + window_size, num_windows_w * window_size)
            
            h_size = h_end - h_start
            w_size = w_end - w_start
            full_attn_map[:, h_start:h_end, w_start:w_end] = window_attn[:, :h_size, :w_size]
    
    full_attn_map = full_attn_map[:, :H, :W]
    
    return full_attn_map.squeeze(0) if B_ == 1 else full_attn_map


class SwinTeacherBackbone(nn.Module):
    def __init__(self, repo_path: str, checkpoint: str = None, model_kwargs: Dict[str, Any] = None,
                 attention_heads_config: Optional[Dict[int, List[Tuple[int, int]]]] = None):

        super().__init__()
        model_kwargs = model_kwargs or {}
        repo_path = Path(repo_path).expanduser().resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        modify_window_attention_to_save_attn()

        from mmseg.models.backbones import SwinTransformer  # pylint: disable=import-error

        adapted_kwargs = model_kwargs.copy()
        if 'in_chans' in adapted_kwargs:
            adapted_kwargs['in_channels'] = adapted_kwargs.pop('in_chans')
        if 'embed_dim' in adapted_kwargs:
            adapted_kwargs['embed_dims'] = adapted_kwargs.pop('embed_dim')
        adapted_kwargs.pop('norm_layer', None)
        if 'ape' in adapted_kwargs:
            adapted_kwargs['use_abs_pos_embed'] = adapted_kwargs.pop('ape')
        adapted_kwargs.pop('use_checkpoint', None)

        self.backbone = SwinTransformer(**adapted_kwargs)
        self.backbone.eval()
        
        self.attention_heads_config = attention_heads_config or {}
        
        self.hooks = []
        self.hook_dict = {}  # {(stage_idx, block_idx, head_idx): AttentionHeadHook}
        
        if attention_heads_config:
            self._register_attention_hooks()
        
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

    def _register_attention_hooks(self):

        for stage_idx, block_head_pairs in self.attention_heads_config.items():
            stage_layer = self.backbone.layers[stage_idx]
            
            for block_idx, head_idx in block_head_pairs:
                if block_idx >= len(stage_layer.blocks):
                    print(f"[Warning] Stage {stage_idx} Block {block_idx} does not exist, skipping")
                    continue
                
                block = stage_layer.blocks[block_idx]
                hook_key = (stage_idx, block_idx, head_idx)
                
                hook = AttentionHeadHook(stage_idx, block_idx, head_idx)
                handle = block.attn.register_forward_hook(hook)
                
                self.hooks.append(handle)
                self.hook_dict[hook_key] = hook

    def _load_checkpoint(self, ckpt_path: str):
        state = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state, dict):
            if 'state_dict' in state:
                state = state['state_dict']
            elif 'model' in state:
                state = state['model']
        new_state = {}
        for key, val in state.items():
            if key.startswith('backbone.'):
                new_state[key[len('backbone.'):]] = val
            elif key.startswith('module.backbone.'):
                new_state[key[len('module.backbone.'):]] = val
            elif key.startswith('model.0.'):  # fallback for timm style
                new_state[key[len('model.0.'):]] = val
        missing, unexpected = self.backbone.load_state_dict(new_state, strict=False)
        print(f"[RelKD] Loaded teacher backbone from {ckpt_path}.")

    def _extract_fused_attention_features(self, x: torch.Tensor) -> List[torch.Tensor]:

        for hook in self.hook_dict.values():
            hook.attention_map = None

        with torch.no_grad():
            stage_features = self.backbone(x)  # List of [B, C, H, W]
        
        if not self.attention_heads_config:
            return [feat.contiguous() for feat in stage_features]
        
        fused_features = []
        window_size = getattr(self.backbone, 'window_size', 7)
        batch_size = x.shape[0]
        
        for stage_idx, original_feat in enumerate(stage_features):
            B, C, H, W = original_feat.shape
            
            if stage_idx not in self.attention_heads_config:

                fused_features.append(original_feat.contiguous())
                continue
            
            attention_maps = []
            
            for block_idx, head_idx in self.attention_heads_config[stage_idx]:
                hook_key = (stage_idx, block_idx, head_idx)
                if hook_key in self.hook_dict:
                    hook = self.hook_dict[hook_key]
                    if hook.attention_map is not None:
                        attn_map = hook.attention_map  # [B_, N, N] where B_ = num_windows * batch_size
                        
                        num_windows_h = (H + window_size - 1) // window_size
                        num_windows_w = (W + window_size - 1) // window_size
                        num_windows = num_windows_h * num_windows_w
                        
                        if attn_map.dim() == 3:
                            B_, N, _ = attn_map.shape
                            
                            attn_per_query = attn_map.mean(dim=2)  
                            
                            if B_ == batch_size * num_windows:
                                attn_reshaped = attn_per_query.view(batch_size, num_windows, N)
                            else:
                                attn_avg = attn_per_query.mean().expand(batch_size, num_windows, N)
                                attn_reshaped = attn_avg.to(attn_per_query.device)
                            
                            attn_windows = attn_reshaped.view(batch_size, num_windows_h, num_windows_w, 
                                                               window_size, window_size)
                            
                            spatial_list = []
                            for h in range(num_windows_h):
                                row_list = []
                                for w in range(num_windows_w):
                                    row_list.append(attn_windows[:, h, w, :, :])
                                spatial_list.append(torch.cat(row_list, dim=2))  
                            spatial_attn = torch.cat(spatial_list, dim=1) 
                            
                            spatial_attn = spatial_attn[:, :H, :W]
                        else:
                            spatial_attn = torch.ones(B, H, W, device=x.device, dtype=x.dtype) / (H * W)
                        
                        attention_maps.append(spatial_attn)
            
            if attention_maps:
                fused_attn = torch.stack(attention_maps).mean(dim=0)  # [B, H, W]
                fused_attn = (fused_attn - fused_attn.min()) / (fused_attn.max() - fused_attn.min() + 1e-8)
                fused_attn = fused_attn.unsqueeze(1).expand_as(original_feat)  # [B, C, H, W]
                fused_feat = original_feat * (1.0 + 0.1 * fused_attn)  
            else:
                fused_feat = original_feat
            
            fused_features.append(fused_feat.contiguous())
        
        return fused_features

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Iterable[torch.Tensor]:

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
        elif x.shape[1] != 3:
            raise ValueError(f"Expected input with 1 or 3 channels, got {x.shape[1]} channels")
        
        if self.attention_heads_config:
            return self._extract_fused_attention_features(x)
        else:
            features = self.backbone(x)
            return [feat.contiguous() for feat in features]
    
    def __del__(self):
        if hasattr(self, 'hooks') and self.hooks:
            for handle in self.hooks:
                handle.remove()


def build_teacher(cfg: Dict[str, Any]) -> nn.Module:
    teacher_type = cfg.get('type', 'swin')
    if teacher_type != 'swin':
        raise ValueError(f"Unsupported teacher type: {teacher_type}")
    repo_path = cfg.get('repo_path')
    if repo_path is None:
        raise ValueError('repo_path must be provided for teacher backbone')
    checkpoint = cfg.get('checkpoint')
    model_kwargs = cfg.get('model_kwargs', {})
    
    attention_heads_config = cfg.get('attention_heads_config', None)

    if attention_heads_config:
        parsed_config = {}
        for stage_str, block_head_list in attention_heads_config.items():
            stage_idx = int(stage_str)
            parsed_list = [(int(b), int(h)) for b, h in block_head_list]
            parsed_config[stage_idx] = parsed_list
        attention_heads_config = parsed_config
    
    teacher = SwinTeacherBackbone(
        repo_path=repo_path,
        checkpoint=checkpoint,
        model_kwargs=model_kwargs,
        attention_heads_config=attention_heads_config
    )
    return teacher

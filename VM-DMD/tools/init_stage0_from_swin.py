import argparse
import torch
import torch.nn as nn

from configs.config_setting import setting_config
from models.vmunet.vmunet import VMUNet
from mmseg.models.backbones import SwinTransformer


class LinearProjectionAdapter:
    
    def __init__(self, teacher_dim, student_dim, init_method='xavier'):

        self.teacher_out, self.teacher_in = teacher_dim
        self.student_out, self.student_in = student_dim
        self.init_method = init_method
        
        if self.teacher_out == self.student_out and self.teacher_in == self.student_in:
            self.projection = None 
        else:

            if self.teacher_out != self.student_out and self.teacher_in == self.student_in:

                self.projection = nn.Parameter(torch.empty(self.student_out, self.teacher_out))
            elif self.teacher_out == self.student_out and self.teacher_in != self.student_in:

                self.projection = nn.Parameter(torch.empty(self.teacher_in, self.student_in))
            else:

                self.proj_out = nn.Parameter(torch.empty(self.student_out, self.teacher_out))
                self.proj_in = nn.Parameter(torch.empty(self.teacher_in, self.student_in))
                self.projection = None
            
            self._initialize_projection()
    
    def _initialize_projection(self):

        if self.projection is not None:
            if self.init_method == 'xavier':
                nn.init.xavier_uniform_(self.projection)
            elif self.init_method == 'kaiming':
                nn.init.kaiming_uniform_(self.projection)
            elif self.init_method == 'identity':

                if self.projection.shape[0] == self.projection.shape[1]:
                    nn.init.eye_(self.projection)
                else:
                    nn.init.xavier_uniform_(self.projection)
            else:  # 'zero'
                nn.init.zeros_(self.projection)
        elif hasattr(self, 'proj_out'):
            nn.init.xavier_uniform_(self.proj_out)
            nn.init.xavier_uniform_(self.proj_in)
    
    def project_weight(self, teacher_weight):

        if self.projection is None:

            return teacher_weight
        
        with torch.no_grad():
            if hasattr(self, 'proj_out'):
                # W_student = W_out @ W_teacher @ W_in^T
                intermediate = self.proj_out @ teacher_weight  # [student_out, teacher_in]
                projected = intermediate @ self.proj_in.T  # [student_out, student_in]
            elif self.teacher_out != self.student_out:
                # W_student = W_proj @ W_teacher
                projected = self.projection @ teacher_weight  # [student_out, teacher_in]
            else:
                # W_student = W_teacher @ W_proj
                # teacher_weight: [teacher_out, teacher_in]
                # self.projection: [teacher_in, student_in]
                # [teacher_out, student_in] = [student_out, student_in] (teacher_out == student_out)
                projected = teacher_weight @ self.projection  # [teacher_out, student_in]
        
        return projected
    
    def project_bias(self, teacher_bias):

        if teacher_bias is None:
            return None
        
        if self.teacher_out == self.student_out:
            return teacher_bias
        
        with torch.no_grad():
            if hasattr(self, 'proj_out'):
                projected = self.proj_out @ teacher_bias.unsqueeze(1)  # [student_out, 1]
                return projected.squeeze(1)  # [student_out]
            elif self.teacher_out != self.student_out:
                projected = self.projection @ teacher_bias.unsqueeze(1)  # [student_out, 1]
                return projected.squeeze(1)  # [student_out]
            else:
                return teacher_bias


def copy_submatrix(dst_linear: torch.nn.Linear, src_linear: torch.nn.Linear):
    """Copy overlapping top-left submatrix from src to dst."""
    with torch.no_grad():
        dst_w = dst_linear.weight
        src_w = src_linear.weight
        min_out = min(dst_w.shape[0], src_w.shape[0])
        min_in = min(dst_w.shape[1], src_w.shape[1])
        dst_w[:min_out, :min_in].copy_(src_w[:min_out, :min_in])

        if dst_linear.bias is not None and src_linear.bias is not None:
            min_bias = min(dst_linear.bias.shape[0], src_linear.bias.shape[0])
            dst_linear.bias[:min_bias].copy_(src_linear.bias[:min_bias])


def partial_copy(src: torch.Tensor, target_rows: int, target_cols: int) -> torch.Tensor:

    out = src.new_zeros(target_rows, target_cols)
    rows = min(target_rows, src.size(0))
    cols = min(target_cols, src.size(1))
    if rows > 0 and cols > 0:
        out[:rows, :cols].copy_(src[:rows, :cols])
    return out


def copy_ss2d_q_o_from_swin(ss2d_module, swin_block):

    if not hasattr(ss2d_module, "in_proj") or not hasattr(ss2d_module, "out_proj"):
        return
    if not hasattr(ss2d_module, "d_inner") or not hasattr(ss2d_module, "d_model"):
        return
    if not hasattr(swin_block, "attn") or not hasattr(swin_block.attn, "w_msa"):
        return

    in_proj: torch.nn.Linear = ss2d_module.in_proj
    out_proj: torch.nn.Linear = ss2d_module.out_proj
    d_inner = ss2d_module.d_inner

    qkv_linear: torch.nn.Linear = swin_block.attn.w_msa.qkv
    with torch.no_grad():
        q_weight, _, _ = torch.chunk(qkv_linear.weight.data, 3, dim=0)
        in_W = in_proj.weight.data
        copied_q = partial_copy(q_weight, d_inner, in_W.size(1))
        in_W[:d_inner, :].copy_(copied_q)

        if in_proj.bias is not None and qkv_linear.bias is not None:
            q_bias, _, _ = torch.chunk(qkv_linear.bias.data, 3, dim=0)
            in_b = in_proj.bias.data
            target_rows = d_inner
            rows = min(target_rows, q_bias.size(0), in_b.size(0))
            if rows > 0:
                in_b[:rows].copy_(q_bias[:rows])

    proj_linear: torch.nn.Linear = swin_block.attn.w_msa.proj
    with torch.no_grad():
        out_W = out_proj.weight.data
        copied_o = partial_copy(proj_linear.weight.data, out_W.size(0), out_W.size(1))
        out_W[:, :].copy_(copied_o)

        if out_proj.bias is not None and proj_linear.bias is not None:
            out_b = out_proj.bias.data
            rows = min(out_b.size(0), proj_linear.bias.data.size(0))
            if rows > 0:
                out_b[:rows].copy_(proj_linear.bias.data[:rows])


def copy_with_projection(dst_linear: torch.nn.Linear, src_linear: torch.nn.Linear, 
                        adapter: LinearProjectionAdapter = None):

    with torch.no_grad():
        if adapter is None:
            copy_submatrix(dst_linear, src_linear)
        else:
            teacher_weight = src_linear.weight  # [teacher_out, teacher_in]
            teacher_bias = src_linear.bias if hasattr(src_linear, 'bias') and src_linear.bias is not None else None
            
            projected_weight = adapter.project_weight(teacher_weight)
            dst_linear.weight.copy_(projected_weight)

            if teacher_bias is not None and dst_linear.bias is not None:
                projected_bias = adapter.project_bias(teacher_bias)
                dst_linear.bias.copy_(projected_bias)
            elif teacher_bias is not None and dst_linear.bias is None:

                pass
            elif teacher_bias is None and dst_linear.bias is not None:
                pass


def build_vmunet():
    cfg = setting_config
    model_cfg = cfg.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=None,
    )
    return model


def build_swin_tiny():
    swin = SwinTransformer(
        pretrain_img_size=224,
        patch_size=4,
        in_channels=3,
        embed_dims=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        strides=(4, 2, 2, 2),
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        with_cp=False,
    )
    return swin


def copy_block(vss_block, swin_block, use_projection=True):

    with torch.no_grad():
        if vss_block.ln_1.weight.shape == swin_block.norm1.weight.shape:
            vss_block.ln_1.weight.copy_(swin_block.norm1.weight)
            vss_block.ln_1.bias.copy_(swin_block.norm1.bias)
        else:
            min_dim = min(vss_block.ln_1.weight.shape[0], swin_block.norm1.weight.shape[0])
            vss_block.ln_1.weight[:min_dim].copy_(swin_block.norm1.weight[:min_dim])
            vss_block.ln_1.bias[:min_dim].copy_(swin_block.norm1.bias[:min_dim])


    self_attn = getattr(vss_block, "self_attention", None)
    if self_attn is not None and hasattr(self_attn, "in_proj") and hasattr(self_attn, "out_proj"):
        copy_ss2d_q_o_from_swin(self_attn, swin_block)
        return

    if use_projection:

        teacher_in_proj_shape = swin_block.attn.w_msa.qkv.weight.shape  # [teacher_out, teacher_in]
        student_in_proj_shape = vss_block.self_attention.in_proj.weight.shape  # [student_out, student_in]
        
        adapter_in_proj = LinearProjectionAdapter(
            teacher_dim=teacher_in_proj_shape,
            student_dim=student_in_proj_shape,
            init_method='xavier'
        )
        
        copy_with_projection(
            vss_block.self_attention.in_proj,
            swin_block.attn.w_msa.qkv,
            adapter_in_proj
        )
    else:
        copy_submatrix(
            vss_block.self_attention.in_proj,
            swin_block.attn.w_msa.qkv
        )
    
    if use_projection:
        teacher_out_proj_shape = swin_block.attn.w_msa.proj.weight.shape  # [teacher_out, teacher_in]
        student_out_proj_shape = vss_block.self_attention.out_proj.weight.shape  # [student_out, student_in]
        
        adapter_out_proj = LinearProjectionAdapter(
            teacher_dim=teacher_out_proj_shape,
            student_dim=student_out_proj_shape,
            init_method='xavier'
        )
        
        copy_with_projection(
            vss_block.self_attention.out_proj,
            swin_block.attn.w_msa.proj,
            adapter_out_proj
        )
    else:
        copy_submatrix(
            vss_block.self_attention.out_proj,
            swin_block.attn.w_msa.proj
        )


def init_stage_from_swin(vmunet: VMUNet, swin: SwinTransformer, stage_idx: int, use_projection=True):

    print(f"Initializing Stage {stage_idx} from Swin (projection: {use_projection})...")
    
    if stage_idx >= len(vmunet.vmunet.layers) or stage_idx >= len(swin.stages):
        print(f"[Warning] Stage {stage_idx} does not exist, skipping initialization")
        return
    
    vss_layer = vmunet.vmunet.layers[stage_idx]
    swin_stage = swin.stages[stage_idx]
    num_blocks = min(len(vss_layer.blocks), len(swin_stage.blocks))
    
    for block_idx in range(num_blocks):
        copy_block(vss_layer.blocks[block_idx], swin_stage.blocks[block_idx], use_projection=use_projection)
    
    print(f"Stage {stage_idx} initialization completed.")


def init_stage0_from_swin(vmunet: VMUNet, swin: SwinTransformer, use_projection=True):
    init_stage_from_swin(vmunet, swin, stage_idx=0, use_projection=use_projection)


def init_stage3_from_swin(vmunet: VMUNet, swin: SwinTransformer, use_projection=True):

    num_stages = len(vmunet.vmunet.layers)
    last_stage_idx = num_stages - 1 
    init_stage_from_swin(vmunet, swin, stage_idx=last_stage_idx, use_projection=use_projection)


def initialize_stage0_with_swin(vmunet: VMUNet, swin_ckpt: str = None, use_projection=True):

    initialize_stages_with_swin(vmunet, swin_ckpt, stages=[0], use_projection=use_projection)


def initialize_stage3_with_swin(vmunet: VMUNet, swin_ckpt: str = None, use_projection=True):

    num_stages = len(vmunet.vmunet.layers)
    last_stage = max(0, num_stages - 1)
    initialize_stages_with_swin(vmunet, swin_ckpt, stages=[last_stage], use_projection=use_projection)


def initialize_stages_with_swin(
    vmunet: VMUNet,
    swin_ckpt: str = None,
    stages: list = None,
    use_projection: bool = True,
):

    swin = build_swin_tiny()
    if swin_ckpt:
        ckpt = torch.load(swin_ckpt, map_location='cpu')
        swin.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)

    total_stages = len(vmunet.vmunet.layers)
    if stages is None:
        stages = list(range(total_stages))

    unique_stages = sorted(set([s for s in stages if 0 <= s < total_stages]))

    for stage_idx in unique_stages:
        init_stage_from_swin(vmunet, swin, stage_idx=stage_idx, use_projection=use_projection)


def main():
    parser = argparse.ArgumentParser(description='Init VM-UNet Stage0 from Swin-Tiny')
    parser.add_argument('--swin-ckpt', type=str, required=False, help='Path to Swin checkpoint (optional)')
    parser.add_argument('--output', type=str, default='stage0_init.pth', help='Output VMUNet checkpoint')
    args = parser.parse_args()

    vmunet = build_vmunet()
    initialize_stage0_with_swin(vmunet, args.swin_ckpt)

    torch.save({'model': vmunet.state_dict()}, args.output)
    print(f'Stage0 weights initialized and saved to {args.output}')


if __name__ == '__main__':
    main()


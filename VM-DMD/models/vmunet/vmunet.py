from .vmamba import VSSM
import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 adapter_dims=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
        self.adapter_dims = adapter_dims
        self.stage_adapters = None
        if adapter_dims is not None:
            adapters = []
            for stage_dim, target_dim in zip(self.vmunet.dims, adapter_dims):
                if stage_dim == target_dim:
                    adapters.append(nn.Identity())
                else:
                    adapters.append(nn.Conv2d(stage_dim, target_dim, kernel_size=1, bias=False))
            if adapters:
                self.stage_adapters = nn.ModuleList(adapters)
    
    def forward(self, x, return_rel_features: bool = False, return_directions: bool = False):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        if return_rel_features or return_directions:
            out = self.vmunet(
                x,
                return_stage_features=True,
                return_directional_features=return_directions,
            )
            if return_directions:
                logits, stage_features, directional_stage_features = out
            else:
                logits, stage_features = out
        else:
            logits = self.vmunet(x)
        if self.num_classes == 1:
            logits = torch.sigmoid(logits)
        if return_rel_features:
            if return_directions:
                return logits, stage_features, directional_stage_features
            return logits, stage_features
        return logits
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            self.vmunet.load_state_dict(model_dict)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            if len(not_loaded_keys) > 0:
                print(f'Encoder: Loaded {len(new_dict)}/{len(model_dict)} weights. ({len(not_loaded_keys)} keys not loaded)')
            else:
                print(f'Encoder: Loaded {len(new_dict)}/{len(model_dict)} weights.')

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            self.vmunet.load_state_dict(model_dict)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            if len(not_loaded_keys) > 0:
                print(f'Decoder: Loaded {len(new_dict)}/{len(model_dict)} weights. ({len(not_loaded_keys)} keys not loaded)')
            else:
                print(f'Decoder: Loaded {len(new_dict)}/{len(model_dict)} weights.')

    def adapt_stage_features(self, stage_features):
        if self.stage_adapters is None:
            return stage_features
        adapted = []
        for feat, adapter in zip(stage_features, self.stage_adapters):
            adapted.append(adapter(feat))
        return adapted
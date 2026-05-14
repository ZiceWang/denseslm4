import torch
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer
from transformers import Mamba2Config
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

config = Mamba2Config(hidden_size=128, state_size=64, num_heads=4, n_groups=1)
mixer = Mamba2Mixer(config, layer_idx=0).cuda().bfloat16()

x = torch.randn(4, 512, 128, device='cuda', dtype=torch.bfloat16)

class Hook:
    @staticmethod
    def forward(module, input, output):
        print(f"in_proj output stride: {output.stride()}")

mixer.in_proj.register_forward_hook(Hook.forward)

mixer(x)


import hashlib
import json
from pathlib import Path
import torch


class LatentMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(12800, 512), torch.nn.GELU(),
            torch.nn.Linear(512, 512), torch.nn.GELU(),
            torch.nn.Linear(512, 8192))

    def forward(self, noisy, context):
        values = torch.cat([noisy.flatten(1), context[:, :, :9, :].flatten(1)], dim=1)
        return self.layers(values).reshape(-1, 32, 16, 16)


def random_base(root, snapshot):
    from diffusers import SanaTransformer2DModel
    config = json.loads((root/'source/mac_sana_pixel_v1.json').read_text())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        model = SanaTransformer2DModel.from_config(json.loads((snapshot/'transformer/config.json').read_text()))
        model.to(dtype=torch.float16).to(dtype=torch.float32)
    hashes = {}
    for name, value in model.state_dict().items():
        assert torch.isfinite(value).all().item()
        hashes[name] = hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
    receipt = root/'random-base-tensors.json'
    if receipt.exists():
        assert json.loads(receipt.read_text()) == hashes
    else:
        from common import atomic_json
        atomic_json(receipt, hashes)
    return model.requires_grad_(False)


def small_model(root):
    config = json.loads((root/'source/mac_sana_pixel_v1.json').read_text())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        model = LatentMLP()
    return model.to('mps'), []

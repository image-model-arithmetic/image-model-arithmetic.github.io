"""A training-only digit loss; it never receives operands or runs in inference."""
import torch
from torch import nn
from torch.nn import functional as F


class DigitCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4))
        self.readout = nn.Linear(64*4*4, 10)

    def forward(self, x):
        return self.readout(self.features(x).flatten(1))


def crops(decoded):
    """Decoded [−1,1] region x=128:384, y=288:512 -> carry then units."""
    assert decoded.ndim == 4 and decoded.shape[1:] == (3, 224, 256)
    gray = (1-decoded.float().mean(1, keepdim=True))/2
    carry = gray[:, :, 32:192, 16:112].contiguous()
    units = gray[:, :, 32:192, 144:240].contiguous()
    return F.interpolate(torch.cat([carry, units], dim=0).contiguous(), size=(32, 32),
                         mode='bilinear', align_corners=False)


def critic_loss(decoded, labels, critic):
    targets = torch.cat([labels//10, labels%10])
    with torch.autocast(device_type=decoded.device.type, enabled=False):
        return F.cross_entropy(critic(crops(decoded)), targets)


def augment(bank, labels, generator):
    """Label-preserving appearance jitter; no expressions or arithmetic labels."""
    count = len(labels)
    angle = (torch.rand(count, generator=generator)-.5)*.18
    scale = .85 + .3*torch.rand(count, generator=generator)
    shifts = (torch.rand(count, 2, generator=generator)-.5)*.16
    matrix = torch.zeros(count, 2, 3)
    matrix[:, 0, 0] = torch.cos(angle)*scale
    matrix[:, 0, 1] = -torch.sin(angle)*scale
    matrix[:, 1, 0] = torch.sin(angle)*scale
    matrix[:, 1, 1] = torch.cos(angle)*scale
    matrix[:, :, 2] = shifts
    selected = bank[labels]
    grid = F.affine_grid(matrix, selected.shape, align_corners=False)
    images = F.grid_sample(selected, grid, align_corners=False, padding_mode='zeros')
    if torch.rand((), generator=generator) < .3:
        images = F.avg_pool2d(F.pad(images.contiguous(), (1, 1, 1, 1)), 3, stride=1)
    images = images + .025*torch.randn(images.shape, generator=generator)
    return images.clamp(0, 1)

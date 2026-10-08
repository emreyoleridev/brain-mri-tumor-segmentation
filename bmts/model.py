"""U-Net (Ronneberger et al., 2015) with batch norm, sized for 256x256 MRI slices."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import normalize, to_tensor


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32, depth=4):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.downs = nn.ModuleList([DoubleConv(in_ch, chs[0])])
        self.downs.extend(DoubleConv(chs[i], chs[i + 1], dropout=0.1 * (i >= depth - 2)) for i in range(depth))
        self.ups = nn.ModuleList(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth)))
        self.dec = nn.ModuleList(DoubleConv(chs[i] * 2, chs[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        self.config = {"in_ch": in_ch, "out_ch": out_ch, "base": base, "depth": depth}

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        skips.pop()
        for up, dec in zip(self.ups, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: Path, device: torch.device | None = None) -> tuple[UNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = UNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device or torch.device("cpu"))
    return model, ckpt.get("meta", {})


@torch.no_grad()
def predict_proba(model: UNet, img_rgb: np.ndarray, tta: bool = True) -> np.ndarray:
    """Tumor probability map (H, W) for one uint8 RGB slice; horizontal-flip TTA averages two passes."""
    device = next(model.parameters()).device
    x = to_tensor(normalize(img_rgb))[None].to(device)
    p = torch.sigmoid(model(x))
    if tta:
        p = (p + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
    return p[0, 0].cpu().numpy()

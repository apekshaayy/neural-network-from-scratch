import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets
from mnist import MEAN, STD

# every function takes a batch of raw pixels in [0, 1], shape (N, 1, 28, 28)
def gaussian_noise(x, std=0.5):
    return (x + std * torch.randn_like(x)).clamp(0, 1)

def rotate(x, deg=40):
    return TF.rotate(x, deg)

def blur(x, sigma = 2.9):
    k = int(2*round(3*sigma)+1)
    return TF.gaussian_blur(x, kernel_size=k, sigma=sigma)

def thicken(x, k=5):                          
    return F.max_pool2d(x, k, stride=1, padding=k // 2)

def thin(x, k=5):                         
    return -F.max_pool2d(-x, k, stride=1, padding=k // 2)

def low_contrast(x, f=0.25):
    return x * f

CORRUPTIONS = {
    "clean": lambda x: x,
    "noise": gaussian_noise,
    "rotate": rotate,
    "blur": lambda x: blur(x, sigma=2.0),
    "thicken": lambda x: thicken(x, k=5),
    "thin": lambda x: thin(x, k=5),
    "low_contrast": lambda x: low_contrast(x, f=0.25),
}

def load_shifted(name, normalize=True):
    """Returns (images, labels) for MNIST test set with one corruption applied."""
    torch.manual_seed(0)             # same noise every run, so results are comparable
    ds = datasets.MNIST("data", train=False, download=True)
    x = ds.data.float().div(255).unsqueeze(1)    # raw pixels in [0, 1]
    x = CORRUPTIONS[name](x)
    if normalize:
        x = (x - MEAN) / STD
    return x, ds.targets


if __name__ == "__main__":           # visual preview
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(CORRUPTIONS), 8, figsize=(10, 9))
    for r, name in enumerate(CORRUPTIONS):
        x, _ = load_shifted(name, normalize=False)
        for c in range(8):
            axes[r, c].imshow(x[c, 0], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_title(name, loc="left", fontsize=9)
    plt.tight_layout()
    plt.savefig("corruptions_preview.png")
    plt.show()
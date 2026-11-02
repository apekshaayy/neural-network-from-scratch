import copy
import torch
import tent                      
import norm          
from mnist import Net
from corruptions import CORRUPTIONS, load_shifted

def load_source():
    model = Net()
    model.load_state_dict(torch.load("mnist_bn.pt"))
    return model.eval()

def make_norm(src):
    return norm.Norm(copy.deepcopy(src))            # deepcopy: Norm edits the model in place

def make_tent(src, lr=1e-3):
    model = tent.configure_model(copy.deepcopy(src))
    tent.check_model(model)
    params, _ = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    return tent.Tent(model, optimizer)

@torch.no_grad()
def accuracy(model, x, y, bs=128):
    correct = 0
    for i in range(0, len(x), bs):
        correct += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(x)

if __name__ == "__main__":
    source = load_source()
    print(f"{'shift':<14}{'source':>8}{'norm':>8}{'tent':>8}")
    for name in CORRUPTIONS:
        x, y = load_shifted(name)
        a_src = accuracy(source, x, y)
        a_norm = accuracy(make_norm(source), x, y)
        a_tent = accuracy(make_tent(source), x, y)   # fresh copy per shift
        print(f"{name:<14}{a_src:>8.2%}{a_norm:>8.2%}{a_tent:>8.2%}")
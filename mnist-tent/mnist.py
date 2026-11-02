import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.flatten(1)
        return self.fc(x)

MEAN, STD = 0.1307, 0.3081

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
])

train_set = datasets.MNIST("data", train=True,  download=True, transform=transform)
test_set  = datasets.MNIST("data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=128, shuffle=False)

@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)

def train(model, loader, epochs=3, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total = 0
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss {total / len(loader):.4f}")

if __name__ == "__main__":
    train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

    model = Net()
    train(model, train_loader)
    torch.save(model.state_dict(), "mnist_bn.pt")
    test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    print(f"clean test accuracy: {accuracy(model, test_loader):.2%}")
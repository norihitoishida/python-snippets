# PyTorch snippets

### CIFAR10 Classifier 
```python
import torch
import torchvision
import torchvision.transforms as transforms

# =============================================================================
# Dataset, DataLoader
# =============================================================================
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4
num_workers = 2
dataroot = "./data"

trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# =============================================================================
# Network
# =============================================================================
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# =============================================================================
# criterion, optimizer, fit
# =============================================================================
import torch.nn.functional as F
import torch.optim as optim

def fit(epochs, model, loss_func, optimizer, train_dl, valid_dl):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    for epoch in range(epochs):
        
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, optimizer)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(f"epoch:{epoch}, val_loss:{val_loss}")

    print('Finished Training')


fit(
    epochs = 2, 
    model = net, 
    loss_func=nn.CrossEntropyLoss(), 
    optimizer=optim.SGD(net.parameters(), lr=0.001, momentum=0.9), 
    train_dl=, 
    valid_dl=
    )



# =============================================================================
# Visualize for computational graph
# =============================================================================
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# =============================================================================
# Visualize for computational graph
# =============================================================================
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# =============================================================================
# fit
# =============================================================================


# =============================================================================
# Visualize for computational graph
# =============================================================================
# =============================================================================
# Visualize for computational graph
# =============================================================================
# =============================================================================
# Visualize for computational graph
# =============================================================================


```
### 
```python

```
### 
```python

```

### 
```python

```
### 
```python

```
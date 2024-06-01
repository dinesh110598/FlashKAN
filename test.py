# %%
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from FlashKAN import FlashKAN
# %%
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# %% 
def test_initialization():
    batch_dim, in_dim, out_dim = 50, 8, 16
    x = torch.rand([batch_dim, in_dim])
    t0, t1, k, G = 0., 5., 4, 20
    t = torch.linspace(t0, t1, G+1)
    t = torch.cat([torch.full([k-1], t0), t,
                    torch.full([k-1], t1)], 0)
    w = torch.zeros([G+k+1, in_dim, out_dim])
    w = torch.nn.init.xavier_normal_(w)
    
    return batch_dim, in_dim, out_dim, t0, t1, k, G, x, t, w

batch_dim, in_dim, out_dim, t0, t1, k, G, x, t, w = test_initialization()
# %%
transform = transforms.Compose(
    [transforms.ToTensor()])

batch = 100
trainset = torchvision.datasets.MNIST("./Data", train=True, download=True,
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./Data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=False)
# %%
net = nn.Sequential(
    nn.Flatten(),
    FlashKAN(28*28, 10, 50, -1., 1.)
).to(device)
net.load_state_dict(torch.load("Saves/mnist_1layer.pth"))
# %%
criterion = nn.CrossEntropyLoss()
metric = lambda out, labels: (torch.argmax(out,1) == labels).float().mean()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# %%
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.
    running_acc = 0.
    for i, data in enumerate(trainloader, 1):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            acc = metric(outputs, labels)

        # print statistics
        running_acc += acc.item()
        running_loss += loss.item()
    print(f'epoch: {epoch + 1}  loss: {running_loss / i:.3f}', 
                f'accuracy: {running_acc / i:.4f}')

print('Finished Training')
# %%
with torch.no_grad():
    running_loss, running_acc = 0., 0.
    for i, data in enumerate(testloader, 1):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        with torch.no_grad():
            acc = metric(outputs, labels)

        # print statistics
        running_acc += acc.item()
        running_loss += loss.item()
    print(f'loss: {running_loss / i:.3f}', 
            f'accuracy: {running_acc / i:.4f}')
# %%
torch.save(net.state_dict(), "Saves/mnist_1layer.pth")
# %%
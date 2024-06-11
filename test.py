# %%
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from layers import FlashKAN, Regular_KAN, KANLinear

from torch.profiler import profile, ProfilerActivity
# %%
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# %% 
def test_initialization():
    batch_dim, in_dim, out_dim = 50, 8, 16
    x = torch.rand([batch_dim, in_dim]).to(device)
    t0, t1, k, G = 0., 5., 4, 20
    t = torch.linspace(t0, t1, G+1)
    t = torch.cat([torch.full([k-1], t0), t,
                    torch.full([k-1], t1)], 0).to(device)
    w = torch.zeros([G+k+1, in_dim, out_dim]).to(device)
    w = torch.nn.init.xavier_normal_(w).to(device)
    
    return batch_dim, in_dim, out_dim, t0, t1, k, G, x, t, w

batch_dim, in_dim, out_dim, t0, t1, k, G, x, t, w = test_initialization()
# %% Profile GPU memory consumption
transform = transforms.ToTensor()

batch = 100
trainset = torchvision.datasets.MNIST("./Data", train=True, download=True,
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./Data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=False)


criterion = nn.CrossEntropyLoss()
metric = lambda out, labels: (torch.argmax(out,1) == labels).float().mean()
# %%
def create_flash_kan(G):
    return nn.Sequential(
        nn.Flatten(),
        FlashKAN(28*28, 10, G)
    ).to(device)
    
def create_reg_kan(G):
    return nn.Sequential(
        nn.Flatten(),
        KANLinear(28*28, 10, G)
    ).to(device)
    
net = create_flash_kan(25)
# %%
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/flashkan_mnist'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:

    for i, data in enumerate(trainloader, 1):
        prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        if i >= 1 + 1 + 3:
            break
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
# %%
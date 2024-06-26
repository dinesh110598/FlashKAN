import torch
from torch import nn
import torch.utils
import torchvision
import torchvision.transforms as transforms
from layers import FlashKAN, Regular_KAN, KANLinear

from time import time
from itertools import product
import os, json
import matplotlib.pyplot as plt

# %matplotlib notebook
# %%
device = torch.device("cuda:0")
# %% MNIST data loaders
transform = transforms.ToTensor()

batch = 200
train_data = torchvision.datasets.MNIST("./Data", train=True, download=True,
                                      transform=transform)
train_data, val_data = torch.utils.data.random_split(train_data, [5/6, 1/6])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch,
                                          shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch,
                                          shuffle=True)


test_data = torchvision.datasets.MNIST(root='./Data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch,
                                         shuffle=False)

criterion = nn.CrossEntropyLoss()
metric = lambda out, labels: (torch.argmax(out,1) == labels).float().mean()
# %% Define model(s)
def create_model(w, G, model_name="flashkan"):
    model_dict = {"flashkan": FlashKAN,
                  "efficientkan": KANLinear}
    KANLayer = model_dict[model_name]
    
    net = nn.Sequential(
        nn.Flatten(),
        KANLayer(28*28, w, G),
        nn.Dropout(0.2),
        KANLayer(w, 10, G)
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    return net, opt
# %%
def train_model(net, opt, epochs=30):
    history = {"epoch": [],
               "time": [],
               "train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []}
    
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.
        running_acc = 0.
        t0 = time()
        for i, data in enumerate(train_loader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            opt.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            with torch.no_grad():
                acc = metric(outputs, labels)
            t1 = time()
            # print statistics
            running_acc += acc.item()
            running_loss += loss.item()
        t1 = time()
        
        
        history["epoch"].append(epoch)
        history["time"].append(t1-t0)
        history["train_loss"].append(running_loss / i)
        history["train_acc"].append(running_acc / i)
        
        running_loss = 0.
        running_acc = 0.
        for j, data in enumerate(val_loader, 1):
            with torch.no_grad():
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                acc = metric(outputs, labels)
                
                running_acc += acc.item()
                running_loss += loss.item()
        
        history["val_loss"].append(running_loss / j)
        history["val_acc"].append(running_acc / j)
    
    return history
# %% Training on a grid of hyperparameters
Gs = [10, 20, 50, 100]
ws = [16, 32, 64]
logs_flashkan = []

for G, w in product(Gs, ws):
    net, opt = create_model(w, G, "flashkan")
    hist = train_model(net, opt, 30)
    logs_flashkan.append(hist)
# %%
logs_effkan = []

for G, w in product(Gs, ws):
    net, opt = create_model(w, G, "efficientkan")
    hist = train_model(net, opt, 30)
    logs_effkan.append(hist)
# %%
with open("./Data/flashkan_grid.json", 'w') as f:
    data = {}
    for i, (G, w) in enumerate(product(Gs, ws)):
        data[f"G_{G}_w_{w}"] = logs_flashkan[i]
    json.dump(data, f)

with open("./Data/effkan_grid.json", 'w') as f:
    data = {}
    for i, (G, w) in enumerate(product(Gs, ws)):
        data[f"G_{G}_w_{w}"] = logs_effkan[i]
    json.dump(data, f)
# %%

    

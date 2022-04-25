from importlib.metadata import requires
from pickletools import optimize
from telnetlib import PRAGMA_HEARTBEAT
import torch
import torch.nn.functional as F 
from torch.autograd import  Variable
import matplotlib.pyplot as plt


torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.05)
    loss_func = torch.nn.MSELoss()
    
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    torch.save(net1, 'net.pkl')    # 保存整个网络
    torch.save(net1.state_dict(), 'net_params.pkl')  # 只保存网络中的参数 (速度快, 占内存少)
    
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    
    
    
def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()
    
save()
restore_net()
restore_params()

    
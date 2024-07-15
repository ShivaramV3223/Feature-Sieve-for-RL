import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics

class net(nn.Module):
    def __init__(self, input_shape, hidden_layers, num_classes, task = "reg"):
        super(net, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.task = task

        self.layers = nn.ModuleList()
        in_dim = input_shape
        for h_dim in hidden_layers:
            self.layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        self.layers.append(nn.Linear(in_dim, num_classes))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        if self.task == 'reg':
            x = self.layers[-1](x)
        else:
            x = F.softmax(self.layers[-1](x), dim = -1)
        return x
        
class Resnet(nn.Module):
    def __init__(self, )

def train_net(model, train_dataloader, val_inputs, val_targets, device, lr = 1e-3, wd = 1e-5, epochs = 10, verbose= False):
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(epochs)):
        tloss = 0
        tloss_num = 0

        for batch_idx, data in enumerate(train_dataloader):
            x, y = data
            x, y = x.to(torch.float32).to(device), y.reshape(-1).to(torch.long).to(device)

            optimizer.zero_grad()
            out = model(x)
            if model.task == "reg":
                loss = F.mse_loss(out, y)
            else:
                loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            tloss += loss.detach().cpu()
            tloss_num += y.shape[0]

        with torch.no_grad():
            inputs, targets = torch.tensor(val_inputs).to(torch.float32).to(device), torch.tensor(val_targets).reshape(-1).to(torch.long).to(device)
            out = model(inputs)
            if model.task == 'reg':
                val_loss = F.mse_loss(out, targets)
            else:
                val_loss = F.cross_entropy(out, targets)
            val_losses.append(val_loss.detach().cpu())
            

        if verbose:
            print(f"Epoch: {epoch} / {epochs} Training Loss: {tloss/ tloss_num} Validation Loss: {val_loss}")
        train_losses.append(tloss/ tloss_num)

    return model, train_losses, val_losses

def eval_net(model, test_data, test_targets, device):
    with torch.no_grad():
        x = torch.tensor(test_data).to(torch.float32).to(device)
        y = torch.tensor(test_targets).reshape(-1).to(torch.long).to(device)

        out = model(x)
        if model.task == 'reg':
            loss = F.mse_loss(out, y).detach().cpu()
        else:
            _, predicted = torch.max(out.detach().cpu(), 1)
            # Update the running total of correct predictions and samples
            loss = metrics.accuracy_score(y.detach().cpu(), predicted)
    return loss
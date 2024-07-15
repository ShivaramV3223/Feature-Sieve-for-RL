import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

SEED = 3407
# setting random seed
torch.cuda.set_device(0)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SiFer_Margin(nn.Module):
    def __init__(self, hidden_layers, aux_layers, aux_position, in_features, out_classes):
        super(SiFer_Margin, self).__init__()
        
        self.hidden_layers = hidden_layers
        self.aux_layers = aux_layers
        self.aux_position = aux_position
        self.in_features = in_features
        self.out_classes = out_classes

        self.layers = nn.ModuleList()
        in_dim = in_features
        for h_dim in hidden_layers:
            self.layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        self.layers.append(nn.Linear(in_dim, out_classes))

        # aux_layers
        self.aux_layers = nn.ModuleList()
        in_dim = hidden_layers[aux_position]
        for h_dim in aux_layers:
            self.aux_layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        self.aux_layers.append(nn.Linear(in_dim, out_classes))

        # parameters
        self.params = nn.ModuleDict({
            "main": self.layers,
            "aux" : self.aux_layers,
            "forget": self.layers[:aux_position + 1]
        })

    def forward(self, x):
        # main network outputs
        out = []
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            out.append(x)
        x = self.layers[-1](x)
        out.append(x)

        # aux network outputs
        sh = out[self.aux_position]
        for aux_layer in self.aux_layers[:-1]:
            sh = F.relu(aux_layer(sh))
        sh = self.aux_layers[-1](sh)

        return x, sh

def learn_main(model, optim_main, x, y):
    model.train()
    optim_main.zero_grad()
    out = model(x)[0]
    loss = F.mse_loss(out, y)
    loss.backward()
    optim_main.step()
    optim_main.zero_grad()
    model.eval()


def learn_aux(model, optim_main, optim_aux, x, y, alpha_aux = 1):
    model.train()
    optim_main.zero_grad()
    aux = model(x)[1]
    loss = alpha_aux * F.mse_loss(aux, y)
    loss.backward()
    optim_aux.step()
    optim_aux.zero_grad()
    model.eval()
    return loss

def forget_aux(model, optim_forget, x, y, margin):
    model.train()
    optim_forget.zero_grad()
    aux = model(x)[1]
    mse_loss = F.mse_loss(aux,y)
    thresh = torch.tensor(-margin).to(torch.float32).to(device)
    loss = torch.max(thresh, -mse_loss)
    loss.backward()
    optim_forget.step()
    optim_forget.zero_grad()
    model.eval()
    return loss

def train_margin(model, train_dataloader, val_inputs, val_targets, device, main_iters = 1, aux_iters = 1, forget_iters = 5, lrs = [1e-3, 1e-3, 1e-4], epochs = 10, verbose = False, margin = 10):
    train_losses = []
    val_losses = []
    steps = 0
    
    optim_main = optim.Adam(model.params.main.parameters(), lr = lrs[0])
    optim_aux = optim.Adam(model.params.aux.parameters(), lr =lrs[1])
    optim_forget = optim.Adam(model.params.aux.parameters(), lr = lrs[2])
    
    for epoch in tqdm(range(epochs)):
        t_loss = 0
        tloss_num = 0
        
        for batch_idx, data in enumerate(train_dataloader):
            x, y = data
            x, y = x.to(torch.float32).to(device), y.reshape(y.shape[0], 1).to(torch.float32).to(device)

            if main_iters and steps % main_iters == 0:
                learn_main(model, optim_main, x, y)
            if aux_iters and steps % aux_iters == 0:
                auxloss = learn_aux(model, optim_main, optim_aux, x, y)
            if forget_iters and steps % forget_iters == 0:
                forget_loss = forget_aux(model, optim_forget, x, y, margin)

            with torch.no_grad():
                out = model(x)[0]
                loss = F.mse_loss(out, y)
                t_loss = loss.detach().cpu()
                tloss_num += y.shape[0]
            steps += 1

        train_losses.append(t_loss / tloss_num)
        
        with torch.no_grad():
            inp, tar = torch.tensor(val_inputs).to(torch.float32).to(device), torch.tensor(val_targets).reshape(val_targets.shape[0], 1).to(torch.float32).to(device)
            out = model(inp)[0]
            val_loss = F.mse_loss(out, tar)
            val_losses.append(val_loss.detach().cpu())
            
        if verbose:
            print(f"Epoch: {epoch} / {epochs} Train Loss: {t_loss / tloss_num} Validation Loss: {val_loss.detach().cpu()}")
    return model, train_losses, val_losses

def eval_fsmodel(model, test_inputs, test_targets, device):
    with torch.no_grad():
        inp, tar = torch.tensor(test_inputs).to(torch.float32).to(device), torch.tensor(test_targets).reshape(test_targets.shape[0], 1).to(torch.float32).to(device)
        out = model(inp)[0]
        loss = F.mse_loss(out, tar).detach().cpu()
    return loss
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
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class SiFer_Ord(nn.Module):
    def __init__(self, hidden_layers, aux_layers, aux_position, in_features, out_classes, num_bins):
        super(SiFer_Ord, self).__init__()
        
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
        self.aux_layers.append(nn.Linear(in_dim, num_bins))

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
        sh = F.sigmoid(sh)

        return x, sh

multi_label_loss = nn.BCEWithLogitsLoss()

def learn_main_ordinal(model, optim_main,x, y):
    model.train()
    optim_main.zero_grad()
    out = model(x)[0]
    loss = F.mse_loss(out, y)
    loss.backward()
    optim_main.step()
    optim_main.zero_grad()
    model.eval()


def learn_aux_ordinal(model, optim_main, optim_aux, x, y_ord, alpha_aux=1):
    model.train()
    optim_main.zero_grad()
    aux = model(x)[1]
    loss = alpha_aux * multi_label_loss(aux, y_ord)
    loss.backward()
    optim_aux.step()
    optim_aux.zero_grad()
    model.eval()
    return loss

def forget_aux_ordinal(model, optim_forget, x,y, num_bins, max_value):
    model.train()
    optim_forget.zero_grad()
    aux = model(x)[1]
    aux_preds = torch.sum(F.sigmoid(aux), dim =-1) * max_value / num_bins
    loss = F.mse_loss(aux_preds.reshape(-1,1), y)
    loss.backward()
    optim_forget.step()
    optim_forget.zero_grad()
    model.eval()
    return loss

def train_ordinal(model, train_dataloader, val_inputs, val_targets, max_value, num_bins = 5, main_iters = 1, aux_iters= 1, forget_iters = 3, lrs = [1e-3, 1e-3, 1e-4], verbose = False, epochs = 10):
    train_losses = []
    val_losses = []
    steps = 0

    optim_main = optim.Adam(model.params.main.parameters(), lr = lrs[0])
    optim_aux = optim.Adam(model.params.aux.parameters(), lr = lrs[1])
    optim_forget = optim.Adam(model.params.forget.parameters(), lr = lrs[2])

    for epoch in tqdm(range(epochs)):
        tloss = 0
        tloss_num = 0

        for batch_idx, data in enumerate(train_dataloader):
            x,y = data
            y_ord = y.reshape(-1).numpy()
            x,y = x.to(torch.float32).to(device), y.reshape(y.shape[0], 1).to(torch.float32).to(device)

            #Ordinal label generation
            y_ord = np.greater_equal.outer(y_ord, np.arange(num_bins)* max_value /num_bins).astype(np.float32)
            y_ord = torch.tensor(y_ord).reshape(y_ord.shape[0], -1).to(torch.float32).to(device)

            if main_iters and steps % main_iters == 0:
                learn_main_ordinal(model, optim_main, x, y)
            if aux_iters and steps % aux_iters == 0:
                aux_loss = learn_aux_ordinal(model, optim_main, optim_aux, x, y_ord)
            if forget_iters and steps % forget_iters == 0:
                forget_loss = forget_aux_ordinal(model, optim_forget, x, y, num_bins, max_value)

            with torch.no_grad():
                out = model(x)[0]
                loss = F.mse_loss(out, y)
                tloss += loss.detach().cpu()
                tloss_num += y.shape[0]
                steps += 1
                
        train_losses.append(tloss / tloss_num)
        with torch.no_grad():
            inp, tar = torch.tensor(val_inputs).to(torch.float32).to(device), torch.tensor(val_targets).reshape(val_targets.shape[0], 1).to(torch.float32).to(device)
            out = model(inp)[0]
            vloss = F.mse_loss(out, tar)
            val_losses.append(vloss)

        if verbose:
            print(f"Epoch: {epoch} / {epochs} Train Loss: {tloss / tloss_num} Validation Loss: {vloss}")
            
    return model, train_losses, val_losses
import torch
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)

""" 
This file contains the training and evaluation functions for the different models
"""
## Forget Loss: Margin Loss
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

## Forgetting Loss: Cross Entropy
def learn_main_ce(FS, optim_main, x, y):
    FS.train()
    optim_main.zero_grad()
    out = FS(x)[0]
    loss = F.mse_loss(out, y)
    loss.backward()
    optim_main.step()
    optim_main.zero_grad()
    FS.eval()

def learn_aux_ce(FS, optim_main, optim_aux, x, y, bins, alpha_aux=1):
    FS.train()
    optim_main.zero_grad()
    aux = FS(x)[1]
    y = torch.bucketize(y,bins, right=True).reshape(-1) - 1
    loss = alpha_aux * F.cross_entropy(aux, y)
    loss.backward()
    optim_aux.step()
    optim_aux.zero_grad()
    FS.eval()
    return loss

def forget_aux_ce(FS, optim_forget, x, num_bins):
    FS.train()
    optim_forget.zero_grad()
    aux = FS(x)[1]
    print(aux)
    print(torch.ones_like(aux) * 1/num_bins)
    loss = F.cross_entropy(aux, torch.ones_like(aux) * 1/num_bins)
    loss.backward()
    optim_forget.step()
    optim_forget.zero_grad()
    FS.eval()
    return loss

# Multilabel Loss function
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
#Training the model
def train(model, train_dataloader, val_dataloader, lr = 0.0005, weight_decay = 0, epochs = 200, verbose = False):
  if verbose:
    print("Training the Model........")

  optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

  losses = []
  val_losses = []
  model.train()

  for epoch in tqdm(range(epochs)):
    tloss = 0
    loss_num = 0

    for batch_idx, data in enumerate(train_dataloader):
      x, y= data
      x = x.to(torch.float32).to(device)
      y = y.reshape(y.shape[0], 1).to(torch.float32).to(device)

      optimizer.zero_grad()
      out = model(x)
      loss = F.mse_loss(out, y)
      loss.backward()
      optimizer.step()

      tloss += loss.detach().cpu()
      loss_num += 1

    with torch.no_grad():
      vloss = 0
      vloss_num = 0

      for batch_idx, data in enumerate(val_dataloader):
        x,y = data
        x = x.to(torch.float32).to(device)
        y = y.reshape(y.shape[0], 1).to(torch.float32).to(device)

        out = model(x)
        loss = F.mse_loss(out, y)

        vloss += loss.detach().cpu()
        vloss_num += 1

    if verbose:
      print(f"Epoch: {epoch}/{epochs} Training loss: {tloss / loss_num} Validation loss: {vloss / vloss_num}")

    losses.append(float(tloss / loss_num))
    val_losses.append(float(vloss / vloss_num))

  return model, losses, val_losses

# Evaluating the model
def eval_model(model, test_dataloader):
  test_loss = 0
  test_loss_num =0

  with torch.no_grad():
    for batch_idx, data in enumerate(test_dataloader):
      x,y = data
      x = x.to(torch.float32).to(device)
      y = y.reshape(y.shape[0], 1).to(torch.float32).to(device)

      out = model(x)
      loss = F.mse_loss(out, y, reduction = 'sum')
        
      test_loss_num += x.shape[0]
      test_loss += loss.detach().cpu()
        
  return float(test_loss / test_loss_num)

def train_fs_mar(model, train_dataloader, val_dataloader, epochs = 100, margin=10, aux_iters=1, main_iters=1, forget_iters=10, verbose=False, lrs=[0.0005, 0.0005, 0.0005], wds = [0,0,0]):
    if verbose:
        print("Training model...............")
        
    train_losses= []
    val_losses = []
    aux_losses = []
    forget_losses = []

    optim_main = optim.Adam(model.params.main.parameters(),lr=lrs[0], weight_decay=wds[0])
    optim_aux = optim.Adam(model.params.aux.parameters(), lr=lrs[1], weight_decay=wds[1])
    optim_forget = optim.Adam(model.params.forget.parameters(), lr=lrs[2], weight_decay=wds[2])

    steps = 0

    for epoch in tqdm(range(epochs)):
        tloss = 0
        loss_num = 0

        # training Part of the code
        for batch_idx, data in enumerate(train_dataloader):
            x,y = data
            x,y = x.to(torch.float32).to(device), y.reshape(y.shape[0], 1).to(torch.float32).to(device)

            if main_iters and steps % main_iters == 0:
                learn_main(model, optim_main, x, y)
            if aux_iters and steps % aux_iters == 0:
                auxloss = learn_aux(model, optim_main, optim_aux, x, y)
            if forget_iters and steps % forget_iters == 0:
                forget_loss = forget_aux(model, optim_forget, x, y, margin)
                forget_losses.append(forget_loss.detach().cpu())

            with torch.no_grad():
                out = model(x)[0]
                loss = F.mse_loss(out, y)

                aux_losses.append(auxloss.detach().cpu())
                
                tloss += loss.detach().cpu()
                loss_num += 1
                steps += 1

        # Validation part of the code
        with torch.no_grad():
            vloss = 0
            vloss_num = 0
            
            for batch_idx, data in enumerate(val_dataloader):
                x,y = data
                x,y = x.to(torch.float32).to(device), y.reshape(y.shape[0], 1).to(torch.float32).to(device)

                out = model(x)[0]
                loss = F.mse_loss(out, y)

                vloss += loss.detach().cpu()
                vloss_num += 1
                
        if verbose:
            print(f"Epochs: {epoch+1}/{epochs} Training Loss: {float(tloss / loss_num)} Validation Loss: {float(vloss / vloss_num)}")
            
        train_losses.append(float(tloss/ loss_num))
        val_losses.append(float(vloss / vloss_num))

    return model, train_losses, val_losses, aux_losses, forget_losses

def eval_FSModel(model, test_dataloader):
    tst_loss = 0
    test_nu = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            x,y= data
            x,y = x.to(torch.float32).to(device), y.reshape(y.shape[0], 1).to(torch.float32).to(device)

            out = model(x)[0]
            loss = F.mse_loss(out, y, reduction = 'sum')

            tst_loss += loss.detach().cpu()
            test_nu += x.shape[0]

    return float(tst_loss / test_nu)

#Functions
def train_fs_ce(model, train_dataloader, val_dataloader, age, num_bins = 20, epochs = 100, margin=10, aux_iters=1, main_iters=1, forget_iters=20, verbose=False, lrs=[0.0005, 0.0005, 0.0005], wds = [0,0,0]):
    if verbose:
        print("Training Model............")
        
    train_losses= []
    val_losses = []
    aux_losses = []
    forget_losses = []

    bin_counts, bin_edges = np.histogram(age, bins = num_bins)
    bin_edges[-1]+=1
    bins = torch.tensor(bin_edges).to(torch.float32).to(device)
    
    optim_main = optim.Adam(model.params.main.parameters(),lr=lrs[0], weight_decay=wds[0])
    optim_aux = optim.Adam(model.params.aux.parameters(), lr=lrs[1], weight_decay=wds[1])
    optim_forget = optim.Adam(model.params.forget.parameters(), lr=lrs[2], weight_decay=wds[2])

    steps = 0

    for epoch in tqdm(range(epochs)):
        tloss = 0
        loss_num = 0

        # Training part of the code
        for batch_idx, data in enumerate(train_dataloader):
            x,y= data
            x,y = x.to(torch.float32).to(device), y.reshape(y.shape[0], 1).to(torch.float32).to(device)

            if main_iters and steps % main_iters == 0:
                learn_main_ce(model, optim_main, x, y)
            if aux_iters and steps % aux_iters == 0:
                aux_loss = learn_aux_ce(model, optim_main, optim_aux, x, y, bins)
            if forget_iters and steps % forget_iters == 0:
                forget_loss = forget_aux_ce(model, optim_forget, x, num_bins)
                forget_losses.append(forget_loss.detach().cpu())

            with torch.no_grad():
                out = model(x)[0]
                loss = F.mse_loss(out, y)

                tloss += loss.detach().cpu()
                loss_num += 1
                steps += 1

            aux_losses.append(aux_loss.detach().cpu())

        # Validation part of the code
        with torch.no_grad():
            vloss = 0
            vloss_num = 0
            
            for batch_idx, data in enumerate(val_dataloader):
                x,y= data
                x,y = x.to(torch.float32).to(device), y.reshape(y.shape[0], 1).to(torch.float32).to(device)

                out = model(x)[0]
                loss = F.mse_loss(out, y)

                vloss += loss.detach().cpu()
                vloss_num += 1
                
        if verbose:
            print(f"Epochs: {epoch}/{epochs} Training Loss: {float(tloss / loss_num)} Validation Loss: {float(vloss / vloss_num)}")
            
        train_losses.append(float(tloss/ loss_num))
        val_losses.append(float(vloss / vloss_num))

    return model, train_losses, val_losses, aux_losses, forget_losses

#Functions
def train_fs_ord(model, train_dataloader, val_dataloader, max_value, num_bins=20, epochs = 100, aux_iters=1, main_iters=1, forget_iters=10, verbose=False, lrs=[0.0005, 0.0005, 0.0005], wds = [0,0,0]):
    train_losses= []
    val_losses = []
    aux_losses = []
    forget_losses = []
    
    optim_main = optim.Adam(model.params.main.parameters(),lr=lrs[0], weight_decay=wds[0])
    optim_aux = optim.Adam(model.params.aux.parameters(), lr=lrs[1], weight_decay=wds[1])
    optim_forget = optim.Adam(model.params.forget.parameters(), lr=lrs[2], weight_decay=wds[2])

    steps = 0

    for epoch in tqdm(range(epochs)):
        tloss = 0
        loss_num = 0

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
                forget_losses.append(forget_loss.detach().cpu())

            with torch.no_grad():
                out = model(x)[0]
                loss = F.mse_loss(out, y)

                tloss += loss.detach().cpu()
                loss_num += 1
                steps += 1

            aux_losses.append(aux_loss.detach().cpu())

        with torch.no_grad():
            vloss = 0
            vloss_num = 0
            
            for batch_idx, data in enumerate(val_dataloader):
                x,y= data
                x,y = x.to(torch.float32).to(device), y.reshape(y.shape[0], 1).to(torch.float32).to(device)

                out = model(x)[0]
                loss = F.mse_loss(out, y)

                vloss += loss.detach().cpu()
                vloss_num += 1
                
        if verbose:
            print(f"Epochs: {epoch}/{epochs} Training Loss: {float(tloss / loss_num)} Validation Loss: {float(vloss / vloss_num)}")
            
        train_losses.append(float(tloss/ loss_num))
        val_losses.append(float(vloss / vloss_num))

    return model, train_losses, val_losses, aux_losses, forget_losses
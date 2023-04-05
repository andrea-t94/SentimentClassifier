import time 
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

# training
def train(model: nn.Module,
          train_loader: DataLoader,
          optimizer: torch.optim,
          lr: torch.float,
          epoch: int,
          device: torch.device):

    log_interval = 5000
    model.train()
    tr_loss=0
    nb_tr_steps=0
    start_time = time.time()
    num_batches = len(train_loader)
    # to store all steps
    loss_steps = []

    for n_batch, batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # CrossEntropy loss is the first output according to HF docs
        loss = outputs[0]
        
        tr_loss += loss.item()
        nb_tr_steps += 1
        if n_batch%log_interval==0:
            loss_step = tr_loss/nb_tr_steps
            loss_steps.append(loss_step)
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            print(f'| epoch {epoch:3d} | {n_batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {loss_step:5.2f}')
            start_time = time.time()
            print(f"Avg Training Loss per {log_interval} steps: {loss_step}")

        loss.backward()
        optimizer.step()
        
    epoch_loss = tr_loss/nb_tr_steps
    return epoch_loss, loss_steps

# model evaluation
def evaluate(model: nn.Module,
            testing_loader: DataLoader,
            device: torch.device):
    model.eval()
    tr_loss=0
    nb_tr_steps=0
    start_time = time.time()
    num_batches = len(testing_loader)
    with torch.no_grad():
        for n_batch, batch in tqdm(enumerate(testing_loader, 0)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # loss (CrossEntropy loss) is the first output according to HF docs
            loss = outputs.loss
            
            tr_loss += loss.item()
            nb_tr_steps += 1
    epoch_loss = tr_loss/nb_tr_steps
    
    return epoch_loss
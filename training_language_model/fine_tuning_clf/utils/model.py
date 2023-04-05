import time 
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import RobertaModel

# model

class RobertaClass(torch.nn.Module):
    def __init__(self, model_path, dropout):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained(model_path, resume_download=True)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

# evaluation
def evaluate(model: nn.Module,
            loss_function: nn.Module,
            testing_loader: DataLoader,
            device: torch.device):
    model.eval()
    start_time = time.time()
    num_batches = len(testing_loader)
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for n_batch, batch in tqdm(enumerate(testing_loader, 0)):
            input_ids = batch['input_ids'].to(device, dtype = torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            outputs = model(input_ids, attention_mask)
            loss = loss_function(outputs, labels)
            tr_loss += loss.item()
            # We take the max logits. If I need probabilities I should use SoftMax final layer only for inferecne
            pred, pred_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(pred_idx, labels)

            nb_tr_steps += 1
            nb_tr_examples+=labels.size(0)

        epoch_loss = tr_loss/nb_tr_steps
        epoch_acc = (n_correct*100)/nb_tr_examples
    
    return epoch_loss, epoch_acc

# training
def train(model: nn.Module,
          loss_function: nn.Module,
          train_loader: DataLoader,
          optimizer: torch.optim,
          lr: torch.float,
          epoch: int,
          device: torch.device):

    log_interval = 5000
    model.train()
    tr_loss=0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    start_time = time.time()
    num_batches = len(train_loader)
    # to store all steps
    loss_steps = []
    acc_steps = []

    for n_batch, batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device, dtype = torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        outputs = model(input_ids, attention_mask=attention_mask)
        # CrossEntropy loss is the first output according to HF docs
        loss = loss_function(outputs, labels)
        tr_loss += loss.item()
        #to get the prediction value and idx
        pred, pred_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(pred_idx, labels)

        nb_tr_steps += 1
        nb_tr_examples+=labels.size(0)

        if n_batch%log_interval==0:
            loss_step = tr_loss/nb_tr_steps
            loss_steps.append(loss_step)
            acc_step = (n_correct*100)/nb_tr_examples 
            acc_steps.append(acc_step)
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            print(f'| epoch {epoch:3d} | {n_batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {loss_step:5.2f} | accuracy {acc_step:5.2f}')
            start_time = time.time()
            print(f"Avg Training Loss per {log_interval} steps: {loss_step}")
            print(f"Avg Training Accuracy per {log_interval} steps: {acc_step}")

        loss.backward()
        optimizer.step()
        
    epoch_loss = tr_loss/nb_tr_steps
    epoch_acc = (n_correct*100)/nb_tr_examples
    
    return epoch_loss, loss_steps, epoch_acc, acc_steps
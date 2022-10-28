import copy
import time
import math
from warnings import warn
from typing import Tuple
import os

# mlflow and S3
import mlflow
import mlflow.pytorch
import boto3

# torch
import torch
from torch import nn, Tensor, device
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer

# custom
from transformer.model import TransformerModel
from transformer.utils import generate_square_subsequent_mask
from datasets.wiki.dataset import WikiVocabulary
from datasets.utils import data_process, batchify, get_batch


# TODO:
#  convert train and eval step into PytorchLighting-ish to add also Mlflow integration
#  checkpointing is done on a service provider (S3)
#  fix dependency: tensorboard 2.10.1 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.1
def train(train_data: Tensor,
          bptt: int, # backpropagation through time, the seq_len fed into the model
          model: nn.Module,
          criterion: nn.Module,
          ntokens: int,
          optimizer: torch.optim,
          scheduler: torch.optim,
          epoch: int,
          device: torch.device) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    # TODO: can be placed outside the training function
    #  since used in both train and evaluate and doesn't need to be recreated every epoch
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)): # looping all over minibatches
        data, targets = get_batch(train_data, i, bptt)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch that can be less then bptt
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(eval_data: Tensor,
             bptt: float,
             model: nn.Module,
             criterion: nn.Module,
             ntokens: int,
             device: torch.device) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


# TODO: make checkpoint dictionary a standard to be used to both checkpoint and load_checkpoint
def checkpoint(model: nn.Module,
               optimizer: torch.optim,
               epoch: int,
               loss: float,
               path: str
               ) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f"The new {path} directory is created!")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, f"{path}model.pt")


def load_checkpoint(path: str) -> Tuple:  # it will become class checkpoint
    checkpoint = torch.load(path)
    loss = checkpoint['loss']
    epoch = checkpoint['epoch']
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']

    return model_state_dict, optimizer_state_dict, epoch, loss



def main():

    # Check that MPS is available
    try:
        device = torch.device(
            'mps' if torch.backends.mps.is_available() else 'cpu')  # 500s for an epoch, at least 10/20x faster
    except:
        warn(f"torch version is {torch.__version__} and doesn't have MPS support, looking dor CUDA")
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"training device: {device}")

    # inputs
    # TODO: shouldn't be embedded here, probably to put on a scheduler input (ZenML?) or as yml
    batch_size = 20  # 64
    eval_batch_size = 10  # 32
    bptt = 35  # 128  # backpropagation through time
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # 8  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    epochs = 5
    n_epochs_to_checkpoint = 5
    model_path = "transformer/artifacts"
    resume_training = False

    # tracking server
    # TODO: embed as env vars in Docker
    # TODO: make it as decision to log metrics on MLflow if users has MLflow instance
    mlflow.set_tracking_uri('http://ec2-13-59-144-81.us-east-2.compute.amazonaws.com')
    mlflow.set_experiment(experiment_name='transformer_sentiment_classification')
    last_run_id = None

    # dataset
    tokenizer = get_tokenizer(tokenizer='basic_english')
    vocab = WikiVocabulary(tokenizer=tokenizer, save_path='datasets/wiki/artifacts').get_vocab()
    train_iter, val_iter, test_iter = WikiText2()
    # batches of shape [seq_len, batch_size]
    train_data = batchify(data_process(train_iter, vocab, tokenizer), batch_size).to(device)
    val_data = batchify(data_process(val_iter, vocab, tokenizer), eval_batch_size).to(device)
    test_data = batchify(data_process(test_iter, vocab, tokenizer), eval_batch_size).to(device)

    # model
    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    # loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # training
    best_val_loss = float('inf')
    best_model = None
    starting_epoch = 1

    # TODO: refactor code to be more concise
    # Please note: we can only resume latest training, both for constraints on training and logging on MLflow run
    if resume_training:
        # last trained model
        model_state_dict, optimizer_state_dict, epoch, val_loss = load_checkpoint(f'{model_path}/latest_model/model.pt')
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        starting_epoch = epoch
        # best trained model
        best_model_state_dict, _, _, best_val_loss = load_checkpoint(f'{model_path}/best_model/model.pt')
        best_model = copy.deepcopy(model).load_state_dict(best_model_state_dict)
        # FIXME
        last_run_id = mlflow.last_active_run().info.run_id  # can only take latest

    # TODO: should be embedded in the same input of the input vars
    params_to_log = {
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'bptt': bptt,
        'emsize': emsize,
        'd_hid': d_hid,
        'nlayers': nlayers,
        'nhead': nhead,
        'dropout': dropout,
        'epochs': epochs,
        'lr': lr,
    'optimizer': optimizer,
    'scheduler': scheduler
    }

    mlflow.start_run(run_id=last_run_id)
    mlflow.log_params(params_to_log)
    for epoch in range(starting_epoch, epochs + 1):
        epoch_start_time = time.time()
        train(train_data, bptt, model, criterion, ntokens, optimizer, scheduler, epoch, device)
        val_loss = evaluate(val_data, bptt, model, criterion, ntokens, device)
        val_ppl = math.exp(val_loss)
        # TODO: add logging on train for train losses
        mlflow.log_metrics({'val_loss': val_loss, 'val_ppl': val_ppl})  # log at each checkpoint
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        # checkpointing
        if epoch % n_epochs_to_checkpoint == 0:
            checkpoint(model, optimizer, epoch, val_loss, f'{model_path}/epoch_{epoch}/')
            checkpoint(model, optimizer, epoch, val_loss, f'{model_path}/latest_model/')
            # TODO: add in documentation both Mlflow and here
            #  add mlflow log model in checkpoint funtion
            # Please note: in order to be able to log the artifacts you must:
            # create ~/.aws/credentials, AWS will look for Default profile. if you don't have it, create it or specify the profile
            # second option (if you don't have the first one) is to specify AWS ACCESS KEY and SECRET ACCESS KEY
            mlflow.pytorch.log_model(model, artifact_path=f'{model_path}/epoch_{epoch}')
            mlflow.pytorch.log_model(model, artifact_path=f'{model_path}/latest_model/')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            print('-' * 89)
            print(f'| checkpointing model at epoch {epoch:3d} |'
                  f'best valid loss {best_val_loss:5.2f} | best valid ppl {math.exp(best_val_loss):8.2f}')
            print('-' * 89)
            checkpoint(best_model, optimizer, epoch, best_val_loss, f'{model_path}/best_model/')
            mlflow.pytorch.log_model(model, artifact_path=f'{model_path}/best_model/')

        scheduler.step()

    # evaluate the best model on the test dataset
    test_loss = evaluate(test_data, bptt, best_model, criterion, ntokens, device)
    test_ppl = math.exp(test_loss)
    mlflow.log_metrics({'test_loss': test_loss, 'test_ppl': test_ppl})
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | '
          f'test ppl {test_ppl:8.2f}')
    print('=' * 89)

if __name__ == '__main__':
    main()
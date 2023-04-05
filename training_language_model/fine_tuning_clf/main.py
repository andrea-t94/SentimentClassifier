import time 
import copy
from tqdm import tqdm
import socket
from datetime import datetime
import os

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.optim import AdamW, Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import RobertaModel, RobertaTokenizer

# custom
from utils.data import SentimentData
from utils.model import RobertaClass, train, evaluate
from params.data import *
from params.model import *


def main():
    
    HF_USER = os.environ.get('HF_USER')
    HF_TOKEN = os.environ.get('HF_TOKEN')
    DIRPATH = os.environ.get('DIRPATH')
    MODEL_VERSION_MLM = os.environ.get('MODEL_VERSION_MLM')

    # data extractionp
    df = pd.read_csv(DATASET_PATH, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)


   # train test split
    train_texts, val_texts, train_sentiment, val_sentiment = train_test_split(df[TEXT_NAME], df[TARGET_NAME], test_size=.2, random_state=10)
    train_texts, test_texts, train_sentiment, test_sentiment = train_test_split(train_texts,  train_sentiment, test_size=.15, random_state=9)

    # tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(f'{HF_USER}/{MODEL_VERSION_MLM}', truncation=False, padding=False, do_lower_case=True)

    # custom DataLoader
    train_loader = DataLoader(SentimentData(train_texts, train_sentiment, tokenizer=tokenizer, max_len=MAX_LEN), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SentimentData(val_texts, val_sentiment, tokenizer=tokenizer, max_len=MAX_LEN), batch_size=EVAL_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SentimentData(test_texts, test_sentiment, tokenizer=tokenizer, max_len=MAX_LEN), batch_size=EVAL_BATCH_SIZE, shuffle=True)
    
    # device used for training
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"training on {device}")

    # model: the fine-tuned on Twitter version
    model =RobertaClass(model_path=f'{HF_USER}/{MODEL_VERSION_MLM}', dropout=DROPOUT)
    model.to(device)

    # Creating the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    if OPTIMIZER_NAME == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER_NAME == 'Adam':
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    else:
        print('No known optimizer has been selected. Options are ...')


    # build tf_log_dir (taken directly from source code)
    run_id = datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()
    tf_log_dir = os.path.join(DIRPATH, f'runs/{params_to_log["model"]}/', run_id)
    # model path
    model_path = os.path.join(DIRPATH, f'models/{params_to_log["model"]}/', run_id)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(tf_log_dir)
    
    # training
    best_val_acc = 0
    best_model = None
    for epoch in range(EPOCHS):
        _, _, train_acc, train_acc_steps = train(
            model, loss_function, train_loader, optimizer, lr=LEARNING_RATE, epoch=epoch, device=device)
        val_loss, val_acc = evaluate(model, loss_function, val_loader, device)
        epoch_start_time = time.time()
        
        # logging on Tensorboard
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        for step, step_acc in enumerate(train_acc_steps):
            print('test', step*epoch)
            writer.add_scalar('Accuracy/train_steps', step_acc, step*epoch)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'| train accuracy {train_acc:5.2f} | valid accuracy {val_acc:5.2f}')
        print('-' * 89)

        # checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            print('-' * 89)
            print(f'| checkpointing best model at epoch {epoch:3d} | best valid accuracy {best_val_acc:5.2f} ')
            print('-' * 89)
            os.makedirs(f'{model_path}/best_model', exist_ok = True) 
            # save everything needed for both inference and resume training
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
                }, os.path.join(f'{model_path}/best_model', 'model_sentiment.tar'))


    test_loss, test_acc = evaluate(best_model,loss_function, test_loader, device)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of training | time: {elapsed:5.2f}s | '
            f'valid accuracy {best_val_acc:5.2f}')
    print('-' * 89)

    writer.add_hparams(params_to_log, {'hparam/test_loss': test_loss,
                                        'hparam/test_acc': test_acc}, run_name='.')
    writer.flush()
    writer.close()

    # save tokenizer vocabulary of my custom model for using for inference
    tokenizer.save_pretrained(f'{model_path}/tokenizer/')



if __name__ == '__main__':
    main()
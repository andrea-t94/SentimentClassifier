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

from transformers import RobertaForMaskedLM, DataCollatorForLanguageModeling
from transformers import RobertaModel, RobertaTokenizer

# custom
from utils.data import group_texts, refactor_dataset, MLMData
from utils.model import train, evaluate
from params.data import *
from params.model import *


def main():
    HF_USER = os.environ.get('HF_USER')
    HF_TOKEN = os.environ.get('HF_TOKEN')
    DIRPATH = os.environ.get('DIRPATH')


    # data extraction
    df = pd.read_csv(DATASET_PATH, encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

    # train test split
    train_texts, val_texts, _, _ = train_test_split(df[TEXT_NAME], df[TEXT_NAME], test_size=.2, random_state=42)
    train_texts, test_texts, _, _ = train_test_split(train_texts, train_texts, test_size=.15, random_state=33)

    # tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base', truncation=False, padding=False, do_lower_case=True)

    # tokenization
    print('tokenization')
    tokenized_train = tokenizer(train_texts.to_list())
    tokenized_val = tokenizer(val_texts.to_list())
    tokenized_test = tokenizer(test_texts.to_list())

    # merging
    print('merge text in chunks')
    lm_train = refactor_dataset(group_texts(tokenized_train, chunk_size=CHUNK_SIZE))
    lm_val = refactor_dataset(group_texts(tokenized_val, chunk_size=CHUNK_SIZE))
    lm_test = refactor_dataset(group_texts(tokenized_test, chunk_size=CHUNK_SIZE))

    # DataLoader w/ DataCollator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    train_loader = DataLoader(MLMData(lm_train, data_collator), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MLMData(lm_val, data_collator), batch_size=EVAL_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(MLMData(lm_test, data_collator), batch_size=EVAL_BATCH_SIZE, shuffle=True)

    # device used for training
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"training on {device}")

    # model
    model = RobertaForMaskedLM.from_pretrained('distilroberta-base')
    model.to(device)
    model.train()

    # Adam with weight decays embedded, lr scheduler?
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
    best_val_loss = float('inf')
    best_model = None
    for epoch in range(EPOCHS):
        train_loss, train_loss_steps = train(
            model, train_loader, optimizer, lr=LEARNING_RATE, epoch=epoch, device=device)
        val_loss = evaluate(model, val_loader, device)
        epoch_start_time = time.time()

        # logging on Tensorboard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        for step, step_loss in enumerate(train_loss_steps):
            writer.add_scalar('Loss/train_steps', step_loss, step*epoch)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'train loss {train_loss:5.2f} | valid loss {val_loss:5.2f}')
        print('-' * 89)
        
        # checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            print('-' * 89)
            print(f'| checkpointing best model at epoch {epoch:3d} | best valid loss {best_val_loss:5.2f} ')
            print('-' * 89)
            best_model.save_pretrained(f'{model_path}/best_model/')
            # save everything else needed for resuming training
            os.makedirs(f'{model_path}/best_model/checkpoint', exist_ok = True) 
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
                }, os.path.join(f'{model_path}/best_model/checkpoint', 'model_sentiment.tar'))
            


    test_loss = evaluate(best_model, test_loader, device)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of training | time: {elapsed:5.2f}s | '
            f'best valid loss {best_val_loss:5.2f}')
    print('-' * 89)

    writer.add_hparams(params_to_log, {'hparam/test_loss': test_loss}, run_name='.')
    writer.flush()
    writer.close()

    # I save the best model on each run and overwrite the existing one. 
    #FIXME: I should manually push a new best model if I find a better one
    best_model.push_to_hub(f'{HF_USER}/{MODEL_VERSION_MLM}', use_auth_token=HF_TOKEN)
    tokenizer.push_to_hub(f'{HF_USER}/{MODEL_VERSION_MLM}', use_auth_token=HF_TOKEN)
    # save tokenizer vocabulary of my custom model for using for inference
    tokenizer.save_pretrained(f'{model_path}/tokenizer/')



if __name__ == '__main__':
    main()
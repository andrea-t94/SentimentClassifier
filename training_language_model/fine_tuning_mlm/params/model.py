import os

DATASET_VERSION = os.environ.get('DATASET_VERSION')
MODEL_VERSION_MLM = os.environ.get('MODEL_VERSION_MLM')

# hparams
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 5e-5
CHUNK_SIZE = 128
OPTIMIZER_NAME = 'AdamW'
SCHEDULER = ''

# params to tfhub
params_to_log = {
    'batch_size': TRAIN_BATCH_SIZE,
    'eval_batch_size': EVAL_BATCH_SIZE,
    'epochs': EPOCHS,
    'learning_rate': LEARNING_RATE,
    'optimizer_name': OPTIMIZER_NAME,
    'chunk_size': CHUNK_SIZE,
    'model': MODEL_VERSION_MLM,
    'dataset': DATASET_VERSION
    #'scheduler': scheduler
}
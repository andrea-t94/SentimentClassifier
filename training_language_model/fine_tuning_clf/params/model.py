import os

DATASET_VERSION = os.environ.get('DATASET_VERSION')
MODEL_VERSION_CLF = os.environ.get('MODEL_VERSION_CLF')

# hparams
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 5e-5
MAX_LEN = 256
DROPOUT = 0.3
OPTIMIZER_NAME = 'Adam'
SCHEDULER = ''


# params to tfhub
params_to_log = {
    'batch_size': TRAIN_BATCH_SIZE,
    'eval_batch_size': EVAL_BATCH_SIZE,
    'epochs': EPOCHS,
    'learning_rate': LEARNING_RATE,
    'optimizer_name': OPTIMIZER_NAME,
    'max_len': MAX_LEN,
    'dropout': DROPOUT,
    'model': MODEL_VERSION_CLF,
    'dataset': DATASET_VERSION
    #'scheduler': scheduler
}

import os

DATASET_VERSION = os.environ.get('DATASET_VERSION')

DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TARGET_NAME = "sentiment"
TEXT_NAME = "text"
DATASET_PATH = f'{DATASET_VERSION}/training.1600000.processed.noemoticon.csv'

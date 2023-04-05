

import pandas as pd
import os

# custom
from params.data import *
from params.model import *

import time


def main():
    #FIXME
    #dirpath = 'data'
    dirpath = os.environ.get('DIRPATH')
    hf_token = os.environ.get('HF_TOKEN')
    model_version_clf = os.environ.get('MODEL_VERSION_CLF')
    print(hf_token)
    print(dirpath)
    df = pd.read_csv('TwitterSentiment140/training.1600000.processed.noemoticon.csv', encoding=DATASET_ENCODING , names=DATASET_COLUMNS)
    print(len(df))
    df = df.head(1000)
    print(f'{dirpath}/models/test.csv')
    df.to_csv(f'{dirpath}/models/test.csv')

if __name__ == '__main__':
    main()
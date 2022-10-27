# from Kaggle Twitter Extractor
# https://www.kaggle.com/code/kaushiksuresh147/twitter-data-extraction-for-ipl2020/notebook


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tweepy as tw # To extarct the twitter data
from tqdm import tqdm


#inputs
consumer_api_key = 'Type your API KEY here '
consumer_api_secret = 'Type your API KEY SECRET here'

auth = tw.OAuthHandler(consumer_api_key, consumer_api_secret)
api = tw.API(auth, wait_on_rate_limit=True)
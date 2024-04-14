import os
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from time import sleep
from enum import Enum

import pandas as pd
import json
import yaml


load_dotenv()
API_KEY = os.environ['FMP_API_KEY']



DATA_DIR = './data'
SOURCE_DIR = './finformer/data'
FMP_DIR = os.path.join(DATA_DIR, 'fmp')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')

os.makedirs(DATASET_DIR, exist_ok=True)

fmp_config_path = os.path.join(SOURCE_DIR, 'fmp-config.yaml')
with open(fmp_config_path, 'r', encoding='utf-8') as file:
    fmp_config = yaml.safe_load(file)

for key in fmp_config.keys():
    fmp_config[key]['dir'] = os.path.join(FMP_DIR, key)
    os.makedirs(fmp_config[key]['dir'], exist_ok=True)


def collect_dataset(key, tickers=None):

    dir = fmp_config[key]['dir']

    if tickers is None:
        path = os.path.join(dir, f'{key}.csv')
        df = pd.read_csv(path)
    else:
        df = None
        for ticker in tickers:
            path = os.path.join(dir, f'{ticker}.csv')
            ticker_df = pd.read_csv(path)
            if df is None:
                df = ticker_df
            else:
                df = pd.concat([df, ticker_df], axis=0)

    dataset_path = os.path.join(DATASET_DIR, key)
    df.to_csv(dataset_path, index=False)

    return df


def get_dataset():

    print(json.dumps(fmp_config, indent=2))

    # Tickers
    tickers_path = os.path.join(FMP_DIR, 'tickers.json')
    tickers = pd.read_json(tickers_path)['ticker'].tolist()

    # Profile
    profile_df = collect_dataset('profile')

    # News
    news_df = collect_dataset('news', tickers)

    # Prices
    prices_df = collect_dataset('prices', tickers)

    # Metrics
    metrics_df = collect_dataset('metrics', tickers)

    return profile_df, news_df, prices_df, metrics_df


if __name__ == '__main__':
    get_dataset()

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


class Dataset:
    def __init__(
        self,
        tickers,
        profile,
        news,
        prices,
        metrics,
    ):
        self.tickers = tickers
        self.profile = profile
        self.news = news
        self.prices = prices
        self.metrics = metrics


def collect_dataset(key):

    config = fmp_config[key]

    dir = config['dir']
    separate = config['separate']

    if separate:
        df = None
        ticker_filename_list = sorted(os.listdir(dir))
        for ticker_filename in ticker_filename_list:
            ticker_path = os.path.join(dir, ticker_filename)
            if os.path.exists(ticker_path) and ticker_path.endswith('.csv'):
                try:
                    ticker_df = pd.read_csv(ticker_path)
                except pd.errors.EmptyDataError:
                    print(f'File {ticker_path} is empty!')
                    continue
                if df is None:
                    df = ticker_df
                else:
                    df = pd.concat([df, ticker_df], axis=0)
    else:
        path = os.path.join(dir, f'{key}.csv')
        df = pd.read_csv(path)

    dataset_path = os.path.join(DATASET_DIR, f'{key}.csv')
    df.to_csv(dataset_path, index=False)

    return df


def get_dataset():

    # Tickers
    tickers_path = os.path.join(FMP_DIR, 'tickers.csv')
    tickers_df = pd.read_csv(tickers_path)['symbol'].unique().tolist()

    # Profile
    profile_df = collect_dataset('profile')

    # News
    news_df = collect_dataset('news')

    # Prices
    prices_df = collect_dataset('prices')

    # Metrics
    metrics_df = collect_dataset('metrics')

    dataset = Dataset(
        tickers=tickers_df,
        profile=profile_df,
        news=news_df,
        prices=prices_df,
        metrics=metrics_df,
    )

    return dataset


if __name__ == '__main__':
    get_dataset()

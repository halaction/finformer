import os
import requests
from dotenv import load_dotenv
import json
import tqdm
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from time import sleep

import pandas as pd


load_dotenv()
API_KEY = os.environ['FMP_API_KEY']

DATA_DIR = './data'

FMP_DIR = os.path.join(DATA_DIR, 'fmp')
SP500_DIR = os.path.join(DATA_DIR, 'sp500')
NASDAQ_DIR = os.path.join(DATA_DIR, 'nasdaq')
os.makedirs(FMP_DIR, exist_ok=True)
os.makedirs(SP500_DIR, exist_ok=True)
os.makedirs(NASDAQ_DIR, exist_ok=True)

fmp_config = {
    'profile': {
        'dir': None,
        'endpoint': 'profile',
        'path_params': {
            'symbol': None,
        },
        'query_params': {
            'apikey': None,
        },
    },
    'news': {
        'dir': None,
        'endpoint': 'stock_news',
        'path_params': None,
        'query_params': {
            'page': None,
            'tickers': None,
            'limit': None,
            'apikey': None,
        },
    },
    'prices': {
        'dir': None,
        'endpoint': 'historical-price-full',
        'path_params': {
            'symbol': None,
        },
        'query_params': {
            'apikey': None,
        },
    },
    'metrics': {
        'dir': None,
        'endpoint': 'key-metrics',
        'path_params': {
            'symbol': None,
        },
        'query_params': {
            'period': None,
            'limit': None,
            'apikey': None,
        },
    },
}

for key in fmp_config.keys():
    fmp_config[key]['dir'] = os.path.join(FMP_DIR, key)
    os.makedirs(fmp_config[key]['dir'], exist_ok=True)

tickers_path = os.path.join(FMP_DIR, 'tickers.json')


def get_url(endpoint, query):

    url = f'https://financialmodelingprep.com/api/v3/{endpoint}/{query}'

    return url


def get_query(path_params, query_params):

    path_list = [
        f'{value}'
        for key, value in path_params.items()
        if value is not None
    ]

    query_list = [
        f'{key}={value}'
        for key, value in query_params.items()
        if value is not None
    ]

    path_str = '/'.join(path_list)
    query_str = '&'.join(query_list)

    query = f'{path_str}?{query_str}'

    return query


def get_data():

    tickers = pd.read_json(tickers_path)['ticker'].tolist()

    for ticker in tickers:

        while True:

            # Profile
            profile_config = fmp_config['profile']

            dir = fmp_config['profile']['dir']
            endpoint = fmp_config['profile']['endpoint']

            path_params = fmp_config['profile']['path_params']
            path_params['symbol'] = ticker

            query_params = fmp_config['profile']['query_params']
            query_params['apikey'] = API_KEY

            query = get_query(path_params, query_params)
            url = get_url(endpoint, query)

            response = requests.get(url)
            data = response.json()

            profile_path = ...

            if True:
                print(f"Success! {data['items']} items retrieved.")

                with open(query_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=2)

                # record[ticker].append(date_interval_str)

                break

            url_profile = ...

            query_news = ...

            query_prices = ...

            query_metrics = ...


if __name__ == '__main__':
    get_data()

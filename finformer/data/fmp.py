import os
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from time import sleep
from enum import Enum

import pandas as pd
import json
import yaml
import functools
import logging


load_dotenv()
API_KEY = os.environ['FMP_API_KEY']

DATA_DIR = './data'
SOURCE_DIR = './finformer/data'

FMP_DIR = os.path.join(DATA_DIR, 'fmp')
SP500_DIR = os.path.join(DATA_DIR, 'sp500')
NASDAQ_DIR = os.path.join(DATA_DIR, 'nasdaq')

os.makedirs(FMP_DIR, exist_ok=True)
os.makedirs(SP500_DIR, exist_ok=True)
os.makedirs(NASDAQ_DIR, exist_ok=True)

fmp_config_path = os.path.join(SOURCE_DIR, 'fmp-config.yaml')
with open(fmp_config_path, 'r', encoding='utf-8') as file:
    fmp_config = yaml.safe_load(file)

for key in fmp_config.keys():
    fmp_config[key]['dir'] = os.path.join(FMP_DIR, key)
    os.makedirs(fmp_config[key]['dir'], exist_ok=True)

tickers_path = os.path.join(FMP_DIR, 'tickers.json')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        output = func(*args, **kwargs)

        endpoint = output.endpoint
        status = output.status

        report = get_report(endpoint, status)
        logger.info(report)

        return output

    return wrapper


class Status(Enum):
    DEFAULT = 0
    SUCCESS = 1
    EXISTS = 2
    FAILED = 3


class Output:
    def __init__(self, endpoint=None, status=None):
        self.endpoint = endpoint
        self.status = status


def get_report(endpoint, status):
    report = f'REPORT (endpoint={endpoint} | status={status.name})'
    return report


def get_url(endpoint, query):
    url = f'https://financialmodelingprep.com/api/v3/{endpoint}{query}'
    return url


def get_query(path_params, query_params):

    if path_params is None:
        path_list = []
    else:
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

    if len(path_list) == 0:
        path_str = ''
    else:
        path_str = '/' + '/'.join(path_list)

    if len(query_list) == 0:
        query_str = ''
    else:
        query_str = '?' + '&'.join(query_list)

    query = path_str + query_str

    return query


def check_tickers(key, tickers, force=False):

    status = Status.DEFAULT
    difference = None

    config = fmp_config[key]

    dir = config['dir']
    separate = config['separate']
    path = os.path.join(dir, f'{key}.csv')

    if not force:
        if separate:
            _tickers_recorded = os.listdir(dir)
            tickers_recorded = []
            for _ticker in _tickers_recorded:
                if _ticker.endswith('.csv'):
                    tickers_recorded.append(_ticker.split('.')[0])
            difference = list(set(tickers) - set(tickers_recorded))

        else:
            if os.path.exists(path):
                data_df = pd.read_csv(path)
                tickers_recorded = data_df['symbol'].tolist()
                difference = list(set(tickers) - set(tickers_recorded))

        if difference is not None:
            if len(difference) == 0:
                status = Status.EXISTS
            else:
                tickers = difference

    return status, tickers


def check_failed_tickers(failed_tickers):

    if len(failed_tickers) > 0:
        status = Status.FAILED

        log_path = os.path.join(FMP_DIR, 'log.json')

        if os.path.exists(log_path):
            with open(log_path, 'w', encoding='utf-8') as file:
                log = json.load(file)
        else:
            log = dict()

        log[key] = {
            'failed': failed_tickers,
        }

        with open(log_path, 'w', encoding='utf-8') as file:
            json.dump(log, file, indent=2)

    else:
        status = Status.SUCCESS

    return status


@log
def get_profile(tickers, force=False, timeout=10):

    key = 'profile'
    key_list = []

    config = fmp_config[key]

    dir = config['dir']
    endpoint = config['endpoint']

    path_params = config['path_params']

    query_params = config['query_params']
    query_params['apikey'] = API_KEY

    status, tickers = check_tickers(key, tickers, force=force)

    if status is not Status.EXISTS:

        retry_tickers = []
        failed_tickers = []

        progress_bar = tqdm(enumerate(tickers), total=len(tickers))
        for i, ticker in progress_bar:
            progress_bar.set_description(desc=f'CALLING (endpoint={endpoint} | ticker={ticker})')

            path_params['symbol'] = ticker

            query = get_query(path_params, query_params)
            url = get_url(endpoint, query)

            try:
                response = requests.get(url, timeout=timeout)
                data = response.json()

                key_list.extend(data)

            except:
                retry_tickers.append(ticker)

            if (i + 1) % 100 == 0:
                sleep(1)

        if len(retry_tickers) > 0:

            progress_bar = tqdm(enumerate(retry_tickers), total=len(retry_tickers))
            for i, ticker in progress_bar:
                progress_bar.set_description(desc=f'RETRYING (endpoint={endpoint} | ticker={ticker})')

                path_params['symbol'] = ticker

                query = get_query(path_params, query_params)
                url = get_url(endpoint, query)

                try:
                    response = requests.get(url, timeout=timeout)
                    data = response.json()

                    key_list.extend(data)

                except:
                    failed_tickers.append(ticker)

                if (i + 1) % 100 == 0:
                    sleep(1)

        key_df = pd.DataFrame(key_list)
        key_path = os.path.join(dir, f'{key}.csv')
        key_df.to_csv(key_path, index=False)

        status = check_failed_tickers(failed_tickers)

    output = Output(
        endpoint=endpoint,
        status=status,
    )

    return output


@log
def get_news(tickers, force=False, timeout=10):

    key = 'news'

    config = fmp_config[key]

    dir = config['dir']
    endpoint = config['endpoint']

    path_params = config['path_params']

    query_params = config['query_params']
    query_params['apikey'] = API_KEY

    status, tickers = check_tickers(key, tickers, force=force)

    if status is not Status.EXISTS:

        retry_tickers = []
        failed_tickers = []

        progress_bar = tqdm(enumerate(tickers), total=len(tickers))
        for i, ticker in progress_bar:

            key_list = []

            query_params['tickers'] = ticker
            page = 0

            while True:
                progress_bar.set_description(desc=f'CALLING (endpoint={endpoint} | ticker={ticker} | page={page})')

                query_params['page'] = page

                query = get_query(path_params, query_params)
                url = get_url(endpoint, query)

                try:
                    response = requests.get(url, timeout=timeout)
                    data = response.json()

                    if len(data) == 0:
                        break
                    else:
                        key_list.extend(data)
                        page += 1

                except:
                    retry_tickers.append(ticker)
                    break

            key_df = pd.DataFrame(key_list)

            key_path = os.path.join(dir, f'{ticker}.csv')
            key_df.to_csv(key_path, index=False)

            if (i + 1) % 100 == 0:
                sleep(1)

        if len(retry_tickers) > 0:

            progress_bar = tqdm(enumerate(retry_tickers), total=len(retry_tickers))
            for i, ticker in progress_bar:

                key_list = []

                query_params['tickers'] = ticker
                page = 0

                while True:
                    progress_bar.set_description(desc=f'RETRYING (endpoint={endpoint} | ticker={ticker} | page={page})')

                    query_params['page'] = page

                    query = get_query(path_params, query_params)
                    url = get_url(endpoint, query)

                    try:
                        response = requests.get(url, timeout=timeout)
                        data = response.json()

                        if len(data) == 0:
                            break
                        else:
                            key_list.extend(data)
                            page += 1

                    except:
                        failed_tickers.append(ticker)
                        break

                key_df = pd.DataFrame(key_list)

                key_path = os.path.join(dir, f'{ticker}.csv')
                key_df.to_csv(key_path, index=False)

                if (i + 1) % 100 == 0:
                    sleep(1)

        status = check_failed_tickers(failed_tickers)

    output = Output(
        endpoint=endpoint,
        status=status,
    )

    return output


@log
def get_prices(tickers, force=False, timeout=10):

    key = 'prices'

    config = fmp_config[key]

    dir = config['dir']
    endpoint = config['endpoint']

    path_params = config['path_params']

    query_params = config['query_params']
    query_params['from'] = '2010-01-01'
    query_params['to'] = '2024-05-01'
    query_params['apikey'] = API_KEY

    status, tickers = check_tickers(key, tickers, force=force)

    if status is not Status.EXISTS:

        retry_tickers = []
        failed_tickers = []

        progress_bar = tqdm(enumerate(tickers), total=len(tickers))
        for i, ticker in progress_bar:
            progress_bar.set_description(desc=f'CALLING (endpoint={endpoint} | ticker={ticker})')

            path_params['symbol'] = ticker

            query = get_query(path_params, query_params)
            url = get_url(endpoint, query)

            try:
                response = requests.get(url, timeout=timeout)
                data = response.json()

                key_list = data['historical']
                key_df = pd.DataFrame(key_list)

                key_path = os.path.join(dir, f'{ticker}.csv')
                key_df.to_csv(key_path, index=False)

            except:
                retry_tickers.append(ticker)

            if (i + 1) % 100 == 0:
                sleep(1)

        if len(retry_tickers) > 0:

            progress_bar = tqdm(enumerate(retry_tickers), total=len(retry_tickers))
            for i, ticker in progress_bar:
                progress_bar.set_description(desc=f'RETRYING (endpoint={endpoint} | ticker={ticker})')

                path_params['symbol'] = ticker

                query = get_query(path_params, query_params)
                url = get_url(endpoint, query)

                try:
                    response = requests.get(url, timeout=timeout)
                    data = response.json()

                    key_list = data['historical']
                    key_df = pd.DataFrame(key_list)

                    key_path = os.path.join(dir, f'{ticker}.csv')
                    key_df.to_csv(key_path, index=False)

                except:
                    failed_tickers.append(ticker)

                if (i + 1) % 100 == 0:
                    sleep(1)

        status = check_failed_tickers(failed_tickers)

    output = Output(
        endpoint=endpoint,
        status=status,
    )

    return output


@log
def get_metrics(tickers, force=False, timeout=10):

    key = 'metrics'

    config = fmp_config[key]

    dir = config['dir']
    endpoint = config['endpoint']

    path_params = config['path_params']

    query_params = config['query_params']
    query_params['period'] = 'quarter'
    query_params['apikey'] = API_KEY

    status, tickers = check_tickers(key, tickers, force=force)

    if status is not Status.EXISTS:

        retry_tickers = []
        failed_tickers = []

        progress_bar = tqdm(enumerate(tickers), total=len(tickers))
        for i, ticker in progress_bar:
            progress_bar.set_description(desc=f'CALLING (endpoint={endpoint} | ticker={ticker})')

            path_params['symbol'] = ticker

            query = get_query(path_params, query_params)
            url = get_url(endpoint, query)

            try:
                response = requests.get(url, timeout=timeout)
                data = response.json()

                key_list = data
                key_df = pd.DataFrame(key_list)

                key_path = os.path.join(dir, f'{ticker}.csv')
                key_df.to_csv(key_path, index=False)

            except:
                retry_tickers.append(ticker)

            if (i + 1) % 100 == 0:
                sleep(1)

        if len(retry_tickers) > 0:

            progress_bar = tqdm(enumerate(retry_tickers), total=len(retry_tickers))
            for i, ticker in progress_bar:
                progress_bar.set_description(desc=f'RETRYING (endpoint={endpoint} | ticker={ticker})')

                path_params['symbol'] = ticker

                query = get_query(path_params, query_params)
                url = get_url(endpoint, query)

                try:
                    response = requests.get(url, timeout=timeout)
                    data = response.json()

                    key_list = data
                    key_df = pd.DataFrame(key_list)

                    key_path = os.path.join(dir, f'{ticker}.csv')
                    key_df.to_csv(key_path, index=False)

                except:
                    failed_tickers.append(ticker)

                if (i + 1) % 100 == 0:
                    sleep(1)

        status = check_failed_tickers(failed_tickers)

    output = Output(
        endpoint=endpoint,
        status=status,
    )

    return output


def get_data(tickers=None, force=False, timeout=10):

    if tickers is None:
        tickers = pd.read_json(tickers_path)['ticker'].tolist()

    get_profile(tickers, force=force, timeout=timeout)
    get_metrics(tickers, force=force, timeout=timeout)
    get_prices(tickers, force=force, timeout=timeout)
    get_news(tickers, force=force, timeout=timeout)


if __name__ == '__main__':
    get_data()

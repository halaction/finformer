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


load_dotenv()
API_KEY = os.environ['FMP_API_KEY']

DATA_DIR = './data'
SOURCE_DIR = './finformer'

SOURCE_DATA_DIR = os.path.join(SOURCE_DIR, 'data')
FMP_DIR = os.path.join(DATA_DIR, 'fmp')
SP500_DIR = os.path.join(DATA_DIR, 'sp500')
NASDAQ_DIR = os.path.join(DATA_DIR, 'nasdaq')
RAW_DATASET_DIR = os.path.join(DATA_DIR, 'raw-dataset')

os.makedirs(FMP_DIR, exist_ok=True)
os.makedirs(SP500_DIR, exist_ok=True)
os.makedirs(NASDAQ_DIR, exist_ok=True)
os.makedirs(RAW_DATASET_DIR, exist_ok=True)

fmp_config_path = os.path.join(SOURCE_DATA_DIR, 'fmp-config.yaml')
with open(fmp_config_path, 'r', encoding='utf-8') as file:
    fmp_config = yaml.safe_load(file)

for key in fmp_config.keys():
    fmp_config[key]['dir'] = os.path.join(FMP_DIR, key)
    os.makedirs(fmp_config[key]['dir'], exist_ok=True)


config_path = os.path.join(SOURCE_DIR, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)


def status_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        output = func(*args, **kwargs)

        endpoint = output.endpoint
        status = output.status

        report = get_report(endpoint, status)
        print(report)

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

    if query_params is None:
        query_list = []
    else:
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


def load_tickers():

    index_names = ['sp500', 'nasdaq']

    df_list = []

    for index_name in index_names:
        endpoint = f'{index_name}_constituent'

        path_params = None
        query_params = {
            'apikey': API_KEY,
        }

        query = get_query(path_params, query_params)

        url = get_url(endpoint, query)

        status = Status.DEFAULT

        try:
            response = requests.get(url)
            data = response.json()

            status = Status.SUCCESS

        except requests.exceptions.Timeout:
            status = Status.FAILED
            break

        df_index = pd.DataFrame(data)
        df_index['index_name'] = index_name

        df_list.append(df_index)

    if status is not Status.FAILED:
        df = pd.concat(df_list, axis=0)

        path = os.path.join(FMP_DIR, 'tickers.csv')
        df.to_csv(path, index=False)

    return status


def load_changes():

    endpoint = 'symbol_change'

    path_params = None
    query_params = {
        'apikey': API_KEY,
    }

    query = get_query(path_params, query_params)
    url = get_url(endpoint, query)

    status = Status.DEFAULT

    try:
        response = requests.get(url)
        data = response.json()

        status = Status.SUCCESS

        df = pd.DataFrame(data)

        # In compliance with collect_key
        df['symbol'] = df['newSymbol']

    except requests.exceptions.Timeout:
        status = Status.FAILED

    if status is not Status.FAILED:
        path = os.path.join(FMP_DIR, 'changes.csv')
        df.to_csv(path, index=False)

    return status


def rename_indicator(row: pd.Series):

    classes = (' A', ' B')
    class_change = row['oldSymbol'].endswith(classes)
    identifier_change = row['oldSymbol'].startswith(row['newSymbol'])

    condition = not (class_change or identifier_change)
    
    return condition


def parse_changes(df_tickers, df_changes, ):

    start_date = pd.to_datetime(config.start_date, format='%Y-%m-%d').date()
    end_date = pd.to_datetime(config.end_date, format='%Y-%m-%d').date()

    columns = df_tickers.columns

    df_tickers['_date_first_added'] = pd.to_datetime(df_tickers['dateFirstAdded'], format='%Y-%m-%d').dt.date
    df_tickers['_delta_first_added'] = end_date - df_tickers['date_first_added']

    condition = (
        (df_tickers['_date_first_added'] <= end_date) 
        & (df_tickers['_delta_first_added'].dt.days > 30)
    )

    df_tickers = df_tickers.loc[condition, columns]
    
    tickers = df_tickers['symbol'].unique().tolist()

    columns = df_changes.columns

    df_changes['_rename_indicator'] = df_changes.apply(rename_indicator, axis=1)
    df_changes['_date'] = pd.to_datetime(df_changes['date'], format='%Y-%m-%d').dt.date

    condition = (
        (df_changes['_date'] >= start_date)  
        & df_changes['newSymbol'].isin(tickers) 
        & (~ df_changes['oldSymbol'].isin(tickers))
        & df_changes['_rename_indicator']
    )

    df_changes = df_changes.loc[condition, columns]

    tickers.extend(df_changes['oldSymbol'].unique().tolist())
    tickers = list(set(tickers))

    changes = dict(zip(df_changes['oldSymbol'], df_changes['newSymbol']))

    return tickers, changes


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


@status_logger
def load_profile(tickers, force=False, timeout=10):

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

        with tqdm(enumerate(tickers), total=len(tickers)) as progress_bar:
            for i, ticker in progress_bar:
                progress_bar.set_description(desc=f'CALLING (endpoint={endpoint} | ticker={ticker})')

                path_params['symbol'] = ticker

                query = get_query(path_params, query_params)
                url = get_url(endpoint, query)

                try:
                    response = requests.get(url, timeout=timeout)
                    data = response.json()

                    key_list.extend(data)

                except requests.exceptions.Timeout:
                    retry_tickers.append(ticker)

                if (i + 1) % 100 == 0:
                    sleep(1)

        if len(retry_tickers) > 0:

            with tqdm(enumerate(retry_tickers), total=len(retry_tickers)) as progress_bar:
                for i, ticker in progress_bar:
                    progress_bar.set_description(desc=f'RETRYING (endpoint={endpoint} | ticker={ticker})')

                    path_params['symbol'] = ticker

                    query = get_query(path_params, query_params)
                    url = get_url(endpoint, query)

                    try:
                        response = requests.get(url, timeout=timeout)
                        data = response.json()

                        key_list.extend(data)

                    except requests.exceptions.Timeout:
                        failed_tickers.append(ticker)

                    if (i + 1) % 100 == 0:
                        sleep(1)

        key_df = pd.DataFrame(key_list)

        key_df.drop_duplicates(inplace=True)

        key_path = os.path.join(dir, f'{key}.csv')
        key_df.to_csv(key_path, index=False)

        status = check_failed_tickers(failed_tickers)

    output = Output(
        endpoint=endpoint,
        status=status,
    )

    return output


@status_logger
def load_news(tickers, force=False, timeout=10):

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

        with tqdm(enumerate(tickers), total=len(tickers)) as progress_bar:
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

                    except requests.exceptions.Timeout:
                        retry_tickers.append(ticker)
                        break

                key_df = pd.DataFrame(key_list)

                key_df.drop_duplicates(inplace=True)

                key_path = os.path.join(dir, f'{ticker}.csv')
                key_df.to_csv(key_path, index=False)

                if (i + 1) % 100 == 0:
                    sleep(1)

        if len(retry_tickers) > 0:

            with tqdm(enumerate(retry_tickers), total=len(retry_tickers)) as progress_bar:
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

                        except requests.exceptions.Timeout:
                            failed_tickers.append(ticker)
                            break

                    key_df = pd.DataFrame(key_list)

                    key_df.drop_duplicates(inplace=True)

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


@status_logger
def load_prices(tickers, force=False, timeout=10):

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

        with tqdm(enumerate(tickers), total=len(tickers)) as progress_bar:
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

                    key_df.drop_duplicates(inplace=True)

                    key_df['symbol'] = ticker

                    key_path = os.path.join(dir, f'{ticker}.csv')
                    key_df.to_csv(key_path, index=False)

                except requests.exceptions.Timeout:
                    retry_tickers.append(ticker)

                if (i + 1) % 100 == 0:
                    sleep(1)

        if len(retry_tickers) > 0:

            with tqdm(enumerate(retry_tickers), total=len(retry_tickers)) as progress_bar:
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

                        key_df.drop_duplicates(inplace=True)

                        key_df['symbol'] = ticker

                        key_path = os.path.join(dir, f'{ticker}.csv')
                        key_df.to_csv(key_path, index=False)

                    except requests.exceptions.Timeout:
                        failed_tickers.append(ticker)

                    if (i + 1) % 100 == 0:
                        sleep(1)

        status = check_failed_tickers(failed_tickers)

    output = Output(
        endpoint=endpoint,
        status=status,
    )

    return output


@status_logger
def load_metrics(tickers, force=False, timeout=10):

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

        with tqdm(enumerate(tickers), total=len(tickers)) as progress_bar:
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

                    key_df.drop_duplicates(inplace=True)

                    key_path = os.path.join(dir, f'{ticker}.csv')
                    key_df.to_csv(key_path, index=False)

                except requests.exceptions.Timeout:
                    retry_tickers.append(ticker)

                if (i + 1) % 100 == 0:
                    sleep(1)

        if len(retry_tickers) > 0:

            with tqdm(enumerate(retry_tickers), total=len(retry_tickers)) as progress_bar:
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

                        key_df.drop_duplicates(inplace=True)

                        key_path = os.path.join(dir, f'{ticker}.csv')
                        key_df.to_csv(key_path, index=False)

                    except requests.exceptions.Timeout:
                        failed_tickers.append(ticker)

                    if (i + 1) % 100 == 0:
                        sleep(1)

        status = check_failed_tickers(failed_tickers)

    output = Output(
        endpoint=endpoint,
        status=status,
    )

    return output


def load_data(force=False, timeout=10):
            
    load_tickers()

    tickers_path = os.path.join(FMP_DIR, 'tickers.csv')
    df_tickers = pd.read_csv(tickers_path)

    load_changes()

    changes_path = os.path.join(FMP_DIR, 'changes.csv')
    df_changes = pd.read_csv(changes_path)

    tickers, changes = parse_changes(df_tickers, df_changes)

    load_profile(tickers, force=force, timeout=timeout)
    load_metrics(tickers, force=force, timeout=timeout)
    load_prices(tickers, force=force, timeout=timeout)
    load_news(tickers, force=force, timeout=timeout)


def collect_key(key, tickers, changes, force=False):

    config = fmp_config[key]

    dir = config['dir']
    separate = config['separate']

    dataset_path = os.path.join(RAW_DATASET_DIR, f'{key}.csv')
    exists = os.path.exists(dataset_path)

    if exists and not force:
        print(f'Dataset {key} exists!')
        df = pd.read_csv(dataset_path)
    else:
        if separate:
            df = None
            ticker_filename_list = sorted(os.listdir(dir))
            with tqdm(ticker_filename_list) as progress_bar:
                for ticker_filename in progress_bar:
                    progress_bar.set_description(desc=f'COLLECTING (key={key} | filename={ticker_filename})')

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

        # Merge ticker changes
        df = df.replace({
            'symbol': changes,
        })

        tickers_merged = list(set(tickers) - set(changes.keys()))
        condition = df['symbol'].isin(tickers_merged)
        df = df.loc[condition]

        df.to_csv(dataset_path, index=False)

    return dataset_path


def collect_data(force=False):

    tickers_path = os.path.join(FMP_DIR, 'tickers.csv')
    df_tickers = pd.read_csv(tickers_path)

    changes_path = os.path.join(FMP_DIR, 'changes.csv')
    df_changes = pd.read_csv(changes_path)

    tickers, changes = parse_changes(df_tickers, df_changes)

    keys = ['profile', 'news', 'prices', 'metrics']
    for key in keys:
        collect_key(key, tickers, changes, force=force)

    tickers_merged = list(set(tickers) - set(changes.keys()))

    condition = df_tickers['symbol'].isin(tickers_merged)
    df_tickers = df_tickers.loc[condition]

    tickers_path = os.path.join(RAW_DATASET_DIR, 'tickers.csv')
    df_tickers.to_csv(tickers_path, index=False)

    condition = df_changes['symbol'].isin(tickers_merged)
    df_changes = df_changes.loc[condition]

    changes_path = os.path.join(RAW_DATASET_DIR, 'changes.csv')
    df_changes.to_csv(changes_path, index=False)

    return RAW_DATASET_DIR

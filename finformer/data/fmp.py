import os
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from time import sleep
from enum import Enum

import pandas as pd
import json

from finformer.utils import FinformerConfig, adaptive_get


load_dotenv()
API_KEY = os.environ['FMP_API_KEY']

config = FinformerConfig()
fmp_config = config.fmp

DATA_DIR = config.dirs.data_dir
SOURCE_DIR = config.dirs.source_dir
SOURCE_DATA_DIR = config.dirs.source_data_dir
FMP_DIR = config.dirs.fmp_dir
RAW_DATASET_DIR = config.dirs.raw_dataset_dir

os.makedirs(FMP_DIR, exist_ok=True)
os.makedirs(RAW_DATASET_DIR, exist_ok=True)

for key in fmp_config.keys():
    os.makedirs(fmp_config[key]['dir'], exist_ok=True)


class Status(Enum):
    DEFAULT = 0
    SUCCESS = 1
    EXISTS = 2
    FAILED = 3


def get_report(endpoint, status):
    report = f'REPORT (endpoint={endpoint} | status={status.name})'
    return report


def get_url(endpoint, query):
    url = f'https://financialmodelingprep.com/api/{endpoint}{query}'
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

    key = 'tickers'
    path = os.path.join(FMP_DIR, f'{key}.csv')

    index_names = ['sp500', 'nasdaq']
    
    df_list = []

    for index_name in index_names:
        endpoint = f'v3/{index_name}_constituent'

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

        report = get_report(endpoint, status)
        print(report)

    if status is not Status.FAILED:
        df = pd.concat(df_list, axis=0)
        df.drop_duplicates(subset='symbol', inplace=True)
        df.to_csv(path, index=False)

    return path


def load_changes():

    key = 'changes'
    path = os.path.join(FMP_DIR, f'{key}.csv')

    endpoint = 'v4/symbol_change'

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
        df.to_csv(path, index=False)

    report = get_report(endpoint, status)
    print(report)

    return path


def check_tickers(key, tickers, force=False):

    status = Status.DEFAULT
    difference = None

    key_config = fmp_config[key]

    dir = key_config['dir']
    separate = key_config['separate']
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
                tickers_recorded = data_df['symbol'].unique().tolist()
                difference = list(set(tickers) - set(tickers_recorded))

        if difference is not None:
            if len(difference) == 0:
                status = Status.EXISTS
            else:
                tickers = difference

    return status, tickers


def check_failed_tickers(key, failed_tickers):

    if len(failed_tickers) > 0:
        status = Status.FAILED

        log_path = os.path.join(FMP_DIR, 'log.json')

        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as file:
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


def load_profile(tickers, force=False, timeout=10):

    key = 'profile'

    key_list = []

    key_config = fmp_config[key]

    dir = key_config['dir']
    endpoint = key_config['endpoint']

    path_params = key_config['path_params']

    query_params = key_config['query_params']
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

                    if len(data) == 0:
                        data = [{'symbol': ticker}, ]

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

                        if len(data) == 0:
                            data = [{'symbol': ticker}, ]

                        key_list.extend(data)

                    except requests.exceptions.Timeout:
                        failed_tickers.append(ticker)

                    if (i + 1) % 100 == 0:
                        sleep(1)

        print(key_list)

        key_df = pd.DataFrame(key_list)

        print(key_df)

        print(len(key_df))

        key_path = os.path.join(dir, f'{key}.csv')
        if os.path.exists(key_path):
            recorded_key_df = pd.read_csv(key_path)
            key_df = pd.concat([key_df, recorded_key_df], axis=0, ignore_index=True)

        print(len(key_df))

        key_df.drop_duplicates(inplace=True)

        print(len(key_df))

        key_df.to_csv(key_path, index=False)

        status = check_failed_tickers(key, failed_tickers)

    report = get_report(endpoint, status)
    print(report)

    return dir


def load_news(tickers, force=False, timeout=10):

    key = 'news'

    key_config = fmp_config[key]

    dir = key_config['dir']
    endpoint = key_config['endpoint']

    path_params = key_config['path_params']

    query_params = key_config['query_params']
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
                        response = adaptive_get(url)
                        data = response.json()

                        if len(data) == 0:
                            break
                        else:
                            key_list.extend(data)
                            page += 1

                    except requests.exceptions.Timeout:
                        retry_tickers.append(ticker)
                        break

                    if (page + 1) % 50 == 0:
                        sleep(5)

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

        status = check_failed_tickers(key, failed_tickers)

    report = get_report(endpoint, status)
    print(report)

    return dir


def load_prices(tickers, force=False, timeout=10):

    key = 'prices'

    key_config = fmp_config[key]

    dir = key_config['dir']
    endpoint = key_config['endpoint']

    path_params = key_config['path_params']

    query_params = key_config['query_params']
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

        status = check_failed_tickers(key, failed_tickers)

    report = get_report(endpoint, status)
    print(report)

    return dir


def load_metrics(tickers, force=False, timeout=10):

    key = 'metrics'

    key_config = fmp_config[key]

    dir = key_config['dir']
    endpoint = key_config['endpoint']

    path_params = key_config['path_params']

    query_params = key_config['query_params']
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

        status = check_failed_tickers(key, failed_tickers)

    report = get_report(endpoint, status)
    print(report)

    return dir


def rename_indicator(row: pd.Series):

    classes = (' A', ' B')
    class_change = row['oldSymbol'].endswith(classes)
    identifier_change = row['oldSymbol'].startswith(row['newSymbol'])

    condition = not (class_change or identifier_change)
    
    return condition


def parse_changes(df_tickers, df_changes, return_df=False):

    start_date = pd.to_datetime(config.params.start_date, format='%Y-%m-%d').date()
    end_date = pd.to_datetime(config.params.end_date, format='%Y-%m-%d').date()

    columns = df_tickers.columns

    df_tickers['_date_first_added'] = df_tickers['dateFirstAdded'].str.strip().combine_first(df_tickers['founded'])
    df_tickers['_date_first_added'] = pd.to_datetime(df_tickers['_date_first_added'], format='%Y-%m-%d').dt.date
    df_tickers['_delta_first_added'] = end_date - df_tickers['_date_first_added']

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

    output = tickers, changes

    if return_df:
        tickers_merged = list(set(tickers) - set(changes.keys()))

        condition = df_tickers['symbol'].isin(tickers_merged)
        df_tickers = df_tickers.loc[condition, :]

        condition = df_changes['symbol'].isin(tickers_merged)
        df_changes = df_changes.loc[condition, :]

        output += df_tickers, df_changes

    return output


def load_data(force=False, timeout=10):

    tickers_path = load_tickers()
    changes_path = load_changes()

    # TODO: Return paths instead of status

    df_tickers = pd.read_csv(tickers_path)
    df_changes = pd.read_csv(changes_path)

    tickers, changes = parse_changes(df_tickers, df_changes, return_df=False)

    load_profile(tickers, force=force, timeout=timeout)
    load_metrics(tickers, force=force, timeout=timeout)
    load_prices(tickers, force=force, timeout=timeout)
    load_news(tickers, force=force, timeout=timeout)

    return FMP_DIR


def collect_key(key, tickers, changes, force=False):

    key_config = fmp_config[key]

    dir = key_config['dir']
    separate = key_config['separate']

    dataset_path = os.path.join(RAW_DATASET_DIR, f'{key}.csv')
    exists = os.path.exists(dataset_path)

    tickers_merged = list(set(tickers) - set(changes.keys()))

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

            # Merge ticker changes
            df = df.replace({
                'symbol': changes,
            })
            
        else:
            path = os.path.join(dir, f'{key}.csv')
            df = pd.read_csv(path)
        
        condition = df['symbol'].isin(tickers_merged)
        df = df.loc[condition]

        df.drop_duplicates(inplace=True)

        df.to_csv(dataset_path, index=False)

    return dataset_path


def collect_data(force=False):

    tickers_path = os.path.join(FMP_DIR, 'tickers.csv')
    df_tickers = pd.read_csv(tickers_path)

    changes_path = os.path.join(FMP_DIR, 'changes.csv')
    df_changes = pd.read_csv(changes_path)

    tickers, changes, df_tickers, df_changes = parse_changes(df_tickers, df_changes, return_df=True)

    tickers_path = os.path.join(RAW_DATASET_DIR, 'tickers.csv')
    df_tickers.to_csv(tickers_path, index=False)

    changes_path = os.path.join(RAW_DATASET_DIR, 'changes.csv')
    df_changes.to_csv(changes_path, index=False)

    keys = ['profile', 'metrics', 'prices', 'news']
    for key in keys:
        collect_key(key, tickers, changes, force=force)

    return RAW_DATASET_DIR

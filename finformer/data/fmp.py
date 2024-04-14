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


class Status(Enum):
    DEFAULT = 0
    SUCCESS = 1
    EXISTS = 2
    FAILED = 3


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


def get_profile(tickers, force=False, timeout=10):

    profile_list = []

    config = fmp_config['profile']

    dir = config['dir']
    endpoint = config['endpoint']

    path_params = config['path_params']

    query_params = config['query_params']
    query_params['apikey'] = API_KEY

    profile_path = os.path.join(dir, 'profile.csv')

    status = Status.DEFAULT

    try:
        if not force:
            if os.path.exists(profile_path):
                profile_df = pd.read_csv(profile_path)
                tickers_recorded = profile_df['symbol'].tolist()
                difference = list(set(tickers) - set(tickers_recorded))
                if len(difference) == 0:
                    status = Status.EXISTS
                    return status
                else:
                    tickers = difference

        retry_tickers = []

        progress_bar = tqdm(enumerate(tickers), total=len(tickers))
        for i, ticker in progress_bar:
            progress_bar.set_description(desc=f'CALL (endpoint={endpoint} | ticker={ticker})')

            path_params['symbol'] = ticker

            query = get_query(path_params, query_params)
            url = get_url(endpoint, query)

            try:
                response = requests.get(url, timeout=timeout)
                data = response.json()

                profile_list.extend(data)

            except requests.exceptions.Timeout:
                retry_tickers.append(ticker)

            if (i + 1) % 100 == 0:
                sleep(1)

        failed_tickers = []

        if len(retry_tickers) > 0:

            progress_bar = tqdm(enumerate(retry_tickers), total=len(retry_tickers))
            for i, ticker in progress_bar:
                progress_bar.set_description(desc=f'RETRY (endpoint={endpoint} | ticker={ticker})')

                path_params['symbol'] = ticker

                query = get_query(path_params, query_params)
                url = get_url(endpoint, query)

                try:
                    response = requests.get(url, timeout=timeout)
                    data = response.json()

                    profile_list.extend(data)

                except requests.exceptions.Timeout:
                    failed_tickers.append(ticker)

        profile_df = pd.DataFrame(profile_list)
        profile_df.to_csv(profile_path, index=False)

        if len(failed_tickers) > 0:
            failed_tickers_dict = {'failed_tickers': failed_tickers}
            failed_tickers_path = os.path.join(dir, 'fail.json')
            with open(failed_tickers_path, 'w', encoding='utf-8') as file:
                json.dump(failed_tickers_dict, file, indent=2)

            status = Status.FAILED
            return status

        else:
            status = Status.SUCCESS
            return status

    finally:
        report = get_report(endpoint, status)
        print(report)


def get_news(tickers, force=False, timeout=10):

    config = fmp_config['news']

    dir = config['dir']
    endpoint = config['endpoint']

    path_params = config['path_params']

    query_params = config['query_params']
    query_params['apikey'] = API_KEY

    status = Status.DEFAULT

    try:
        if not force:
            _tickers_recorded = os.listdir(dir)
            tickers_recorded = []

            for _ticker in _tickers_recorded:
                if _ticker[-4:] == '.csv':
                    tickers_recorded.append(_ticker.split('.')[0])

            difference = list(set(tickers) - set(tickers_recorded))

            if len(difference) == 0:
                status = Status.EXISTS
                return status
            else:
                tickers = difference

        retry_tickers = []

        progress_bar = tqdm(enumerate(tickers), total=len(tickers))
        for i, ticker in progress_bar:

            news_list = []

            query_params['tickers'] = ticker
            page = 0

            while True:
                progress_bar.set_description(desc=f'CALL (endpoint={endpoint} | ticker={ticker} | page={page})')

                query_params['page'] = page

                query = get_query(path_params, query_params)
                url = get_url(endpoint, query)

                try:
                    response = requests.get(url, timeout=timeout)
                    data = response.json()

                    if len(data) == 0:
                        break
                    else:
                        news_list.extend(data)
                        page += 1

                except requests.exceptions.Timeout:
                    retry_tickers.append(ticker)
                    break

            news_df = pd.DataFrame(news_list)

            news_path = os.path.join(dir, f'{ticker}.csv')
            news_df.to_csv(news_path, index=False)

            if (i + 1) % 100 == 0:
                sleep(1)

        failed_tickers = []

        if len(retry_tickers) > 0:

            progress_bar = tqdm(enumerate(retry_tickers), total=len(retry_tickers))
            for i, ticker in progress_bar:
                progress_bar.set_description(desc=f'RETRY (endpoint={endpoint} | ticker={ticker})')

                news_list = []

                query_params['tickers'] = ticker
                page = 0

                while True:
                    query_params['page'] = page

                    query = get_query(path_params, query_params)
                    url = get_url(endpoint, query)

                    try:
                        response = requests.get(url, timeout=timeout)
                        data = response.json()

                        if len(data) == 0:
                            break
                        else:
                            news_list.extend(data)
                            page += 1

                    except requests.exceptions.Timeout:
                        failed_tickers.append(ticker)
                        break

                news_df = pd.DataFrame(news_list)

                news_path = os.path.join(dir, f'{ticker}.csv')
                news_df.to_csv(news_path, index=False)

                if (i + 1) % 100 == 0:
                    sleep(1)

        if len(failed_tickers) > 0:
            failed_tickers_dict = {'failed_tickers': failed_tickers}
            failed_tickers_path = os.path.join(dir, 'fail.json')
            with open(failed_tickers_path, 'w', encoding='utf-8') as file:
                json.dump(failed_tickers_dict, file, indent=2)

            status = Status.FAILED
            return status

        else:
            status = Status.SUCCESS
            return status

    finally:
        report = get_report(endpoint, status)
        print(report)


def get_prices(tickers, force=False, timeout=10):

    config = fmp_config['prices']

    dir = config['dir']
    endpoint = config['endpoint']

    path_params = config['path_params']

    query_params = config['query_params']
    query_params['from'] = '2010-01-01'
    query_params['to'] = '2024-05-01'
    query_params['apikey'] = API_KEY

    status = Status.DEFAULT

    try:
        if not force:
            _tickers_recorded = os.listdir(dir)
            tickers_recorded = []

            for _ticker in _tickers_recorded:
                if _ticker[-4:] == '.csv':
                    tickers_recorded.append(_ticker.split('.')[0])

            difference = list(set(tickers) - set(tickers_recorded))

            if len(difference) == 0:
                status = Status.EXISTS
                return status
            else:
                tickers = difference

        retry_tickers = []

        progress_bar = tqdm(enumerate(tickers), total=len(tickers))
        for i, ticker in progress_bar:
            progress_bar.set_description(desc=f'CALL (endpoint={endpoint} | ticker={ticker})')

            path_params['symbol'] = ticker

            query = get_query(path_params, query_params)
            url = get_url(endpoint, query)

            try:
                response = requests.get(url, timeout=timeout)
                data = response.json()

                prices_list = data['historical']
                prices_df = pd.DataFrame(prices_list)

                prices_path = os.path.join(dir, f'{ticker}.csv')
                prices_df.to_csv(prices_path, index=False)

            except requests.exceptions.Timeout:
                retry_tickers.append(ticker)

            if (i + 1) % 100 == 0:
                sleep(1)

        failed_tickers = []

        if len(retry_tickers) > 0:

            progress_bar = tqdm(enumerate(retry_tickers), total=len(retry_tickers))
            for i, ticker in progress_bar:
                progress_bar.set_description(desc=f'RETRY (endpoint={endpoint} | ticker={ticker})')

                path_params['symbol'] = ticker

                query = get_query(path_params, query_params)
                url = get_url(endpoint, query)

                try:
                    response = requests.get(url, timeout=timeout)
                    data = response.json()

                    prices_list = data['historical']
                    prices_df = pd.DataFrame(prices_list)

                    prices_path = os.path.join(dir, f'{ticker}.csv')
                    prices_df.to_csv(prices_path, index=False)

                except requests.exceptions.Timeout:
                    failed_tickers.append(ticker)

                if (i + 1) % 100 == 0:
                    sleep(1)

        if len(failed_tickers) > 0:
            failed_tickers_dict = {'failed_tickers': failed_tickers}
            failed_tickers_path = os.path.join(dir, 'fail.json')

            with open(failed_tickers_path, 'w', encoding='utf-8') as file:
                json.dump(failed_tickers_dict, file, indent=2)

            status = Status.FAILED
            return status

        else:
            status = Status.SUCCESS
            return status

    finally:
        report = get_report(endpoint, status)
        print(report)


def get_metrics(tickers, force=False, timeout=10):

    config = fmp_config['metrics']

    dir = config['dir']
    endpoint = config['endpoint']

    path_params = config['path_params']

    query_params = config['query_params']
    query_params['period'] = 'quarter'
    query_params['apikey'] = API_KEY

    status = Status.DEFAULT

    try:
        if not force:
            _tickers_recorded = os.listdir(dir)
            tickers_recorded = []

            for _ticker in _tickers_recorded:
                if _ticker[-4:] == '.csv':
                    tickers_recorded.append(_ticker.split('.')[0])

            difference = list(set(tickers) - set(tickers_recorded))

            if len(difference) == 0:
                status = Status.EXISTS
                return status
            else:
                tickers = difference

        retry_tickers = []

        progress_bar = tqdm(enumerate(tickers), total=len(tickers))
        for i, ticker in progress_bar:
            progress_bar.set_description(desc=f'CALL (endpoint={endpoint} | ticker={ticker})')

            path_params['symbol'] = ticker

            query = get_query(path_params, query_params)
            url = get_url(endpoint, query)

            try:
                response = requests.get(url, timeout=timeout)
                data = response.json()

                metrics_list = data
                metrics_df = pd.DataFrame(metrics_list)

                metrics_path = os.path.join(dir, f'{ticker}.csv')
                metrics_df.to_csv(metrics_path, index=False)

            except requests.exceptions.Timeout:
                retry_tickers.append(ticker)

            if (i + 1) % 100 == 0:
                sleep(1)

        failed_tickers = []

        if len(retry_tickers) > 0:

            progress_bar = tqdm(enumerate(retry_tickers), total=len(retry_tickers))
            for i, ticker in progress_bar:
                progress_bar.set_description(desc=f'RETRY (endpoint={endpoint} | ticker={ticker})')

                path_params['symbol'] = ticker

                query = get_query(path_params, query_params)
                url = get_url(endpoint, query)

                try:
                    response = requests.get(url, timeout=timeout)
                    data = response.json()

                    metrics_list = data
                    metrics_df = pd.DataFrame(metrics_list)

                    metrics_path = os.path.join(dir, f'{ticker}.csv')
                    metrics_df.to_csv(metrics_path, index=False)

                except requests.exceptions.Timeout:
                    failed_tickers.append(ticker)

                if (i + 1) % 100 == 0:
                    sleep(1)

        if len(failed_tickers) > 0:
            failed_tickers_dict = {'failed_tickers': failed_tickers}
            failed_tickers_path = os.path.join(dir, 'fail.json')

            with open(failed_tickers_path, 'w', encoding='utf-8') as file:
                json.dump(failed_tickers_dict, file, indent=2)

            status = Status.FAILED
            return status

        else:
            status = Status.SUCCESS
            return status

    finally:
        report = get_report(endpoint, status)
        print(report)


def get_data(tickers=None, force=False, timeout=10):

    if tickers is None:
        tickers = pd.read_json(tickers_path)['ticker'].tolist()

    get_profile(tickers, force=force, timeout=timeout)
    get_metrics(tickers, force=force, timeout=timeout)
    get_prices(tickers, force=force, timeout=timeout)
    get_news(tickers, force=force, timeout=timeout)

if __name__ == '__main__':
    get_data()

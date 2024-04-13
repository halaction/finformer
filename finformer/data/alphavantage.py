import os
import requests
from dotenv import load_dotenv
import json
import tqdm
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from time import sleep


load_dotenv()

API_KEYS = os.environ['ALPHAVANTAGE_API_KEYS'].split(',')
API_KEY_IDX = 0

DATA_DIR = './data'
ALPHAVANTAGE_DIR = os.path.join(DATA_DIR, 'alphavantage')
NASDAQ_DIR = os.path.join(DATA_DIR, 'nasdaq')

os.makedirs(ALPHAVANTAGE_DIR, exist_ok=True)
os.makedirs(NASDAQ_DIR, exist_ok=True)


def get_query(params):

    query_list = [
        f'{key}={value}'
        for key, value in params.items()
        if value is not None
    ]

    query = '&'.join(query_list)

    return query


def main():

    print(API_KEYS)
    print(os.getcwd())
    print(os.listdir())

    tickers_dir = os.path.join(ALPHAVANTAGE_DIR, 'tickers')

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(os.path.join(NASDAQ_DIR, 'tickers_nasdaq.json'), 'r', encoding='utf-8') as file:
        tickers = json.load(file)['tickers']

    year_start = 2022
    year_end = 2024

    date_start = date(year_start, 1, 1)
    date_end = date(year_end, 1, 1)

    query_delta = relativedelta(years=1)
    n_queries = year_end - year_start

    date_schedule = [
        (date_start + query_delta * i, min(date_start + query_delta * (i + 1) - timedelta(days=1), date_end))
        for i in range(n_queries)
    ]

    print(date_schedule)

    # TODO: Possible multiprocessing?

    for ticker in tickers:

        ticker_dir = os.path.join(tickers_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        sleep(1)
        print()

        for query_date_start, query_date_end in date_schedule:

            query_date_start_str = query_date_start.strftime('%Y%m%d')
            query_date_end_str = query_date_end.strftime('%Y%m%d')
            date_interval_str = f'{query_date_start_str}-{query_date_end_str}'
            query_path = os.path.join(ticker_dir, f'{date_interval_str}.json')

            print(f'ticker: {ticker}')
            print(f'date_interval_str: {date_interval_str}')

            # if date_interval_str in record[ticker]:
            if os.path.exists(query_path):
                print('Query has already been completed. Continuing...')
                continue

            while True:

                # News sentiment: https://www.alphavantage.co/documentation/#news-sentiment
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': ticker,
                    'time_from': f"{query_date_start_str}T0000",
                    'time_to': f"{query_date_end_str}T0000",
                    'limit': 1000,
                    'apikey': API_KEYS[API_KEY_IDX],
                }

                query = get_query(params)
                url = f'https://www.alphavantage.co/query?{query}'
                response = requests.get(url)
                data = response.json()

                print(f'url: {url}')

                if 'items' in data:
                    print(f"Success! {data['items']} items retrieved.")

                    with open(query_path, 'w', encoding='utf-8') as file:
                        json.dump(data, file, indent=2)

                    # record[ticker].append(date_interval_str)

                    break

                if 'Information' in data:
                    if 'time range' in data['Information']:
                        print('Failed! No data for the selected time range. Continuing...')
                        break
                    else:
                        print('Failed! Max number of requests reached. Trying another API key...')
                        API_KEY_IDX += 1

                if API_KEY_IDX >= len(API_KEYS) - 1:
                    print('Failed! All API keys were tried. Finishing...')

                    # with open(record_path, 'w', encoding='utf-8') as file:
                    #     json.dump(record, file, indent=2)

                    return None


if __name__ == '__main__':
    main()

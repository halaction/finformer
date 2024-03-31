import os
import requests
from dotenv import load_dotenv
import json
import tqdm
from datetime import date, timedelta


load_dotenv()


def get_query(params):

    query_list = [
        f'{key}={value}'
        for key, value
        in params.items()
        if value is not None
    ]

    query = '&'.join(query_list)

    return query


if __name__ == '__main__':

    if not os.path.exists('./data'):
        os.mkdir('./data')

    with open('./data/tickers.json', 'r', encoding='utf-8') as file:
        tickers = json.load(file)['tickers']

    date_start = date(year='')
    date_end =

    time_partition =

    # TODO: Possible multiprocessing?

    for ticker in tickers:

        for

        # News sentiment: https://www.alphavantage.co/documentation/#news-sentiment
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'time_from': '20230101T0000',
            'time_to': '20240101T0000',
            'limit': 1000,
            'apikey': os.environ['ALPHAVANTAGE_KEY'],
        }

        query = get_query(params)
        print(query)
        url = f'https://www.alphavantage.co/query?{query}'
        response = requests.get(url)
        data = response.json()

        print(json.dumps(data, indent=2))

        with open('./data/data.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)

        # with open('./data/data.json', 'r', encoding='utf-8') as file:
        #     data = json.load(file)

        #print(json.dumps(data, indent=4))

        #output = data['feed']

        #print(output)


        # Fundamental data: https://www.alphavantage.co/documentation/#fundamentals



import os
import requests
from dotenv import load_dotenv
import json
import pandas as pd

from tqdm import tqdm


load_dotenv()

DATA_PATH = './data'

source_news_path = os.path.join(DATA_PATH, 'analyst_ratings_processed.csv')
news_path = os.path.join(DATA_PATH, 'data-news.csv')
news_tickers_path = os.path.join(DATA_PATH, 'data-news-tickers.json')

source_prices_path = os.path.join(DATA_PATH, 'archive-prices')
prices_path = os.path.join(DATA_PATH, 'data-prices.csv')
prices_tickers_path = os.path.join(DATA_PATH, 'data-prices-tickers.json')

news_intersection_path = os.path.join(DATA_PATH, 'data-news-intersection.csv')
prices_intersection_path = os.path.join(DATA_PATH, 'data-prices-intersection.csv')
intersection_tickers_path = os.path.join(DATA_PATH, 'data-intersection-tickers.json')


def get_data_news(min_count: int = 300):

    os.makedirs(DATA_PATH, exist_ok=True)

    df_news_original = pd.read_csv(source_news_path)

    df_news = (
        df_news_original
        .drop(columns=[
            'Unnamed: 0',
        ])
        .rename(columns={
            'date': 'timestamp',
            'stock': 'ticker',
        })
        .dropna()
    )

    counts = df_news['ticker'].value_counts()
    tickers = counts[counts > min_count].index
    df_news = df_news[df_news.ticker.isin(tickers)]

    df_news['date'] = df_news['timestamp'].str[:10]
    df_news = df_news.loc[:, ['ticker', 'date', 'timestamp', 'title']]
    df_news.to_csv(news_path, index=False)

    df_news_tickers = tickers.to_frame(index=False, name='tickers')
    df_news_tickers.to_json(news_tickers_path)

    return df_news, df_news_tickers


def get_data_prices():

    os.makedirs(DATA_PATH, exist_ok=True)

    df_prices = None
    filename_list = os.listdir(source_prices_path)

    for filename in tqdm(filename_list):

        if filename[-4:] != '.csv':
            print(filename)
            continue

        ticker = filename.split('.')[0]
        file_path = os.path.join(source_prices_path, filename)

        df_ticker = pd.read_csv(file_path)
        df_ticker['ticker'] = ticker

        if df_prices is None:
            df_prices = df_ticker
        else:
            df_prices = pd.concat([df_prices, df_ticker], axis=0)

    tickers = pd.Series(df_prices['ticker'].unique())
    df_prices_tickers = tickers.to_frame(name='tickers')
    df_prices_tickers.to_json(prices_tickers_path)

    df_prices.to_csv(prices_path, index=False)

    return df_prices, df_prices_tickers


def get_data():

    os.makedirs(DATA_PATH, exist_ok=True)

    if not (os.path.exists(news_path) and os.path.exists(news_tickers_path)):
        df_news, df_news_tickers = get_data_news()
    else:
        df_news = pd.read_csv(news_path)
        df_news_tickers = pd.read_json(news_tickers_path)

    if not (os.path.exists(prices_path) and os.path.exists(prices_tickers_path)):
        df_prices, df_prices_tickers = get_data_prices()
    else:
        df_prices = pd.read_csv(prices_path)
        df_prices_tickers = pd.read_json(prices_tickers_path)

    prices_tickers = set(df_prices_tickers.values.flatten().tolist())
    news_tickers = set(df_news_tickers.values.flatten().tolist())

    intersection_tickers = prices_tickers & news_tickers

    df_prices_intersection = df_prices[df_prices['ticker'].isin(intersection_tickers)]
    df_news_intersection = df_news[df_news['ticker'].isin(intersection_tickers)]

    tickers = pd.Series(list(intersection_tickers))
    df_tickers_intersection = tickers.to_frame(name='tickers')
    df_tickers_intersection.to_json(intersection_tickers_path)

    df_prices_intersection.to_csv(prices_intersection_path, index=False)
    df_news_intersection.to_csv(news_intersection_path, index=False)

    return df_prices_intersection, df_news_intersection, df_tickers_intersection

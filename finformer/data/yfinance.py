import os
import requests
from dotenv import load_dotenv
import json
import pandas as pd

from tqdm import tqdm
import yfinance


load_dotenv()

DATA_DIR = './data'
KAGGLE_DIR = os.path.join(DATA_DIR, 'kaggle')
YFINANCE_DIR = os.path.join(DATA_DIR, 'yfinance')

os.makedirs(KAGGLE_DIR, exist_ok=True)
os.makedirs(YFINANCE_DIR, exist_ok=True)

source_news_path = os.path.join(KAGGLE_DIR, 'archive-news/analyst_ratings_processed.csv')
news_path = os.path.join(KAGGLE_DIR, 'data-news.csv')
news_tickers_path = os.path.join(KAGGLE_DIR, 'data-news-tickers.json')

prices_yf_path = os.path.join(YFINANCE_DIR, 'data-prices-yf.csv')
info_yf_path = os.path.join(YFINANCE_DIR, 'data-info-yf.json')
prices_yf_tickers_path = os.path.join(YFINANCE_DIR, 'data-prices-yf-tickers.json')


def get_data_prices():

    df_news = pd.read_csv(news_path)
    df_news_tickers = pd.read_json(news_tickers_path)

    date_start = df_news['date'].min()
    date_end = df_news['date'].max()

    tickers_list = df_news_tickers['tickers'].to_list()

    info = dict()
    df_prices_yf = None

    for ticker in tqdm(tickers_list):

        tickers_pointer = yfinance.Ticker(ticker)

        info[ticker] = tickers_pointer.info

        history = tickers_pointer.history(
            start=date_start,
            end=date_end,
        )

        if len(history) == 0:
            continue

        history = history.reset_index()

        history['ticker'] = ticker
        history['date'] = history['Date'].dt.strftime('%Y-%m-%d')
        history = history.drop(columns=['Date', ])

        if df_prices_yf is None:
            df_prices_yf = history
        else:
            df_prices_yf = pd.concat([df_prices_yf, history], axis=0)

    tickers = pd.Series(df_prices_yf['ticker'].unique())
    df_prices_yf_tickers = tickers.to_frame(name='tickers')
    df_prices_yf_tickers.to_json(prices_yf_tickers_path)

    df_prices_yf.to_csv(prices_yf_path, index=False)

    with open(info_yf_path, 'w', encoding='utf-8') as file:
        json.dump(info, file, indent=2)

    return df_prices_yf, info_yf_path, df_prices_yf_tickers


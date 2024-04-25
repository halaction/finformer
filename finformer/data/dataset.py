import os
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from time import sleep
from enum import Enum

import pandas as pd
import json
import yaml
import re

from huggingface_hub import login, hf_hub_download
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.preprocessing import LabelEncoder


load_dotenv()
HF_TOKEN = os.environ['HF_TOKEN']

login(token=HF_TOKEN)

DATA_DIR = './data'
SOURCE_DIR = './finformer/data'

fmp_config_path = os.path.join(SOURCE_DIR, 'fmp-config.yaml')
with open(fmp_config_path, 'r', encoding='utf-8') as file:
    fmp_config = yaml.safe_load(file)


def snake_case(string: str):
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    string = pattern.sub('_', string).lower()
    return string


def get_tickers():

    repo_id = 'halaction/finformer-data'
    filename = 'dataset/tickers.csv'

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset')
    df = pd.read_csv(path)

    tickers = df['symbol'].unique().tolist()

    return tickers


class SentimentDataset(Dataset):

    def __init__(self, tickers):
        self.tickers = tickers
        self.keys = ['news']

        repo_id = 'halaction/finformer-data'

        for key in self.keys:
            filename = f'dataset/{key}.csv'

            path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset')
            df = pd.read_csv(path)

            df = df[df['symbol'].isin(self.tickers)]

            setattr(self, key, df)

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class TimeSeriesDataset(Dataset):

    def __init__(self, tickers):

        self.tickers = tickers
        self.keys = ['prices', 'metrics', 'profile']

        repo_id = 'halaction/finformer-data'

        for key in self.keys:

            filename = f'dataset/{key}.csv'

            path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset')
            df = pd.read_csv(path)

            df = df[df['symbol'].isin(self.tickers)]

            setattr(self, key, df)

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class FinformerConfig:
    context_length = 30
    prediction_length = 15
    start_date = '2020-01-01'
    end_date = '2024-04-15'


class FinformerDataset(Dataset):

    def __init__(self, config):

        self.config = config

        self.tickers = self.get_tickers()

        # TODO: Encode all indices before sampling

        self.df_news = self.get_news()
        self.df_prices = self.get_prices()
        self.df_profile = self.get_profile()
        self.df_metrics = self.get_metrics()

        self.df_calendar = self.get_calendar()
        self.df_report_calendar = self.get_report_calendar()

    def get_news(self):

        key = 'news'

        filename = f'dataset/{key}.csv'
        path = hf_hub_download(repo_id=self.config.repo_id, filename=filename, repo_type='dataset')
        df = pd.read_csv(path)

        df.drop_duplicates(inplace=True)

        condition = df['symbol'].isin(self.tickers)
        columns = ['symbol', 'publishedDate', 'title', 'text']

        df = df.loc[condition, columns]

        df = df.rename(columns={
            'symbol': 'ticker',
            'publishedDate': 'timestamp',
        })

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        df['date'] = df['timestamp'].dt.date

        level = ['ticker', 'timestamp']
        df.set_index(level, inplace=True)
        df.sort_index(level=level, ascending=True, inplace=True)

        return df

    def get_prices(self):

        key = 'prices'

        filename = f'dataset/{key}.csv'
        path = hf_hub_download(repo_id=self.config.repo_id, filename=filename, repo_type='dataset')
        df = pd.read_csv(path)

        df.drop_duplicates(inplace=True)

        condition = df['symbol'].isin(self.tickers)
        columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']

        df = df.loc[condition, columns]

        df = df.rename(columns={
            'symbol': 'ticker',
            'date': 'timestamp',
        })

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        df['date'] = df['timestamp'].dt.date

        df = df.drop(columns=['timestamp', ])

        start = pd.to_datetime(self.config.start_date, format='%Y-%m-%d')
        end = pd.to_datetime(self.config.end_date, format='%Y-%m-%d')
        date_index = pd.date_range(start=start, end=end, freq='D')

        levels = ['ticker', 'date']
        df.set_index(levels, inplace=True)
        df.sort_index(level=levels, ascending=True, inplace=True)

        df = df.loc[pd.IndexSlice[:, start:end], :]

        index = pd.MultiIndex.from_product(
            [df.index.get_level_values('ticker').unique(), date_index],
            names=levels,
        )

        df = df.reindex(index)

        return df

    def get_profile(self):

        key = 'profile'

        filename = f'dataset/{key}.csv'
        path = hf_hub_download(repo_id=self.config.repo_id, filename=filename, repo_type='dataset')
        df = pd.read_csv(path)

        df.drop_duplicates(inplace=True)

        # TODO: Think about what you can do with description w/ or w/o LLM

        condition = df['symbol'].isin(self.tickers)
        columns = ['symbol', 'industry', 'sector', 'country', 'ipoDate']

        df = df.loc[condition, columns]

        df = df.rename(columns={
            'symbol': 'ticker',
            'ipoDate': 'date_ipo',
        })

        df['date_ipo'] = pd.to_datetime(df['date_ipo'], format='%Y-%m-%d')
        df['_max_date_ipo'] = df['date_ipo'].max()

        df['age_ipo'] = df['_max_date_ipo'] - df['date_ipo']

        df = df.drop(columns=['date_ipo', '_max_date_ipo'])

        # Hard-coding missing values
        df.loc[df['ticker'] == 'SOLV', 'sector'] = 'Healthcare'
        df.loc[df['ticker'] == 'SOLV', 'industry'] = 'Medical - Healthcare Information Services'
        df.loc[df['country'].isna(), 'country'] = 'US'

        levels = ['ticker', ]
        df.set_index(levels, inplace=True)
        df.sort_index(level=levels, ascending=True, inplace=True)

        columns = ['sector', 'industry', 'country']
        label_encoders = {column: LabelEncoder() for column in columns}

        for column, encoder in label_encoders.items():
            df[f'{column}_id'] = encoder.fit_transform(df[column])

        df = df.drop(columns=columns)

        return df

    def get_metrics(self):

        key = 'metrics'

        filename = f'dataset/{key}.csv'
        path = hf_hub_download(repo_id=self.config.repo_id, filename=filename, repo_type='dataset')
        df = pd.read_csv(path)

        df.drop_duplicates(inplace=True)

        condition = df['symbol'].isin(self.tickers)
        columns = [
            'symbol',
            'date',
            'calendarYear',
            'period',
            'revenuePerShare',
            'netIncomePerShare',
            'marketCap',
            'peRatio',
            'priceToSalesRatio',
            'pocfratio',
            'pfcfRatio',
            'pbRatio',
            'ptbRatio',
            'debtToEquity',
            'debtToAssets',
            'currentRatio',
            'interestCoverage',
            'incomeQuality',
            'salesGeneralAndAdministrativeToRevenue',
            'researchAndDdevelopementToRevenue',
            'intangiblesToTotalAssets',
            'capexToOperatingCashFlow',
            'capexToDepreciation',
            'investedCapital',
        ]

        df = df.loc[condition, columns]

        df = df.rename(columns={
            **{column: snake_case(column) for column in columns},
            'symbol': 'ticker',
        })

        df['report_period'] = df['calendar_year'].astype(str).str.cat(df['period'], sep='-')

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['start_date'] = df['date']

        levels = ['ticker', 'report_period']
        df.set_index(levels, inplace=True)
        df.sort_index(level=levels, ascending=True, inplace=True)

        df['end_date'] = df.groupby(level=['ticker', ])['start_date'].shift(periods=-1, fill_value=self.config.end_date)
        df['end_date'] = df['end_date'] - pd.to_timedelta(1, unit='D')

        df = df.drop(columns=[
            'date',
            'calendar_year',
            'period',
        ])

        return df

    def get_calendar(self):

        start = pd.to_datetime(self.config.start_date, format='%Y-%m-%d')
        end = pd.to_datetime(self.config.end_date, format='%Y-%m-%d')
        date_index = pd.date_range(start=start, end=end, freq='D')

        df = pd.DataFrame(date_index, columns=['date'])

        df['weekday'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month_name()
        df['days_in_month'] = df['date'].dt.days_in_month
        df['is_month_end'] = df['date'].dt.is_month_end
        df['quarter'] = df['date'].dt.quarter
        df['is_quarter_end'] = df['date'].dt.is_quarter_end
        start_year = pd.to_datetime(self.config.start_date, format='%Y-%m-%d').year
        df['age'] = df['date'].dt.year - start_year

        columns = ['weekday', 'month', 'quarter']
        df = pd.get_dummies(df, columns=columns, drop_first=True)

        return df

    def get_report_calendar(self):

        columns = ['start_date', 'end_date']
        df = self.metrics[columns]

        return df

    def get_past_values(self, ticker_index, date_index):
        columns = ['open', 'close']
        df_past_values = self.df_prices.loc[pd.IndexSlice[ticker_index, date_index], columns]

        shape = len(ticker_index), len(date_index), len(columns)
        past_values = torch.tensor(df_past_values.values, dtype=torch.float64).view(shape)

        return past_values

    def get_past_observed_mask(self, past_values):
        past_observed_mask = past_values.isnan()
        return past_observed_mask

    def get_static_categorical_features(self, ticker_index):
        columns = ['sector_id', 'industry_id', 'country_id']
        df_static_categorical_features = self.df_profile.loc[ticker_index, columns]

        static_categorical_features = torch.tensor(df_static_categorical_features.values, dtype=torch.int64)

        return static_categorical_features

    def get_static_real_features(self, ticker_index, report_index):
        index = zip(ticker_index, report_index)
        df_static_real_features = self.df_metrics.loc[index, :]

        # [B, ]
        static_real_features = torch.tensor(df_static_real_features.values, dtype=torch.float64)

        return static_real_features

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass



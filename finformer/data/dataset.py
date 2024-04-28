import os
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import torch
import yaml
import re
from itertools import product

from huggingface_hub import login, hf_hub_download
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from finformer.utils import FinformerConfig, snake_case


load_dotenv()
HF_TOKEN = os.environ['HF_TOKEN']

login(token=HF_TOKEN)

config = FinformerConfig()

DATA_DIR = './data'
SOURCE_DIR = './finformer'

RAW_DATASET_DIR = os.path.join(DATA_DIR, 'raw-dataset')


class FinformerData:

    def __init__(self, config, force=False):

        self.config = config

        self.start_date = pd.to_datetime(self.config.start_date, format='%Y-%m-%d').date()
        self.end_date = pd.to_datetime(self.config.end_date, format='%Y-%m-%d').date()
        self.date_index = pd.date_range(start=self.start_date, end=self.end_date, freq='D')

        self.keys = ['tickers', 'news', 'prices', 'profile', 'metrics', 'calendar']

        os.makedirs('./data', exist_ok=True)

        self.load(force=force)
        self.save(force=force)

    def load(self, force=False):

        for key in tqdm(self.keys):

            path = f'./data/{key}.pkl'
            exists = os.path.exists(path)

            if exists and not force:
                df = pd.read_pickle(path)
            else:
                get_data = getattr(self, f'get_{key}')
                df = get_data()

            setattr(self, key, df)

    def save(self, force=False):

        for key in tqdm(self.keys):

            path = f'./data/{key}.pkl'
            exists = os.path.exists(path)

            if not exists or force:
                df = getattr(self, key)
                df.to_pickle(f'./data/{key}.pkl')

    def get_tickers(self):

        repo_id = 'halaction/finformer-data'
        filename = 'dataset/tickers.csv'

        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset')
        df = pd.read_csv(path)

        tickers = pd.Series(df['symbol'].unique())

        return tickers

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

        condition_date = (df['date'] >= self.start_date) & (df['date'] <= self.end_date)
        df = df.loc[condition_date, :]

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

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
        df['date'] = df['timestamp'].dt.date

        condition_date = (df['date'] >= self.start_date) & (df['date'] <= self.end_date)
        df = df.loc[condition_date, :]

        df = df.drop(columns=['timestamp', ])

        levels = ['ticker', 'date']
        df.set_index(levels, inplace=True)
        df.sort_index(level=levels, ascending=True, inplace=True)

        df = df.loc[pd.IndexSlice[:, self.start_date:self.end_date], :]

        index = pd.MultiIndex.from_product(
            [df.index.get_level_values('ticker').unique(), self.date_index],
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

        df['age_ipo'] = (df['_max_date_ipo'] - df['date_ipo']).dt.days

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

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date
        df['start_date'] = df['date']

        levels = ['ticker', 'date']
        df.set_index(levels, inplace=True)
        df.sort_index(level=levels, ascending=True, inplace=True)

        df['end_date'] = df.groupby(level=['ticker', ])['start_date'].shift(periods=-1, fill_value=self.end_date)
        df['end_date'] = df['end_date'] - pd.to_timedelta(1, unit='D')

        condition_date = (df['end_date'] >= self.start_date) & (df['start_date'] <= self.end_date)
        df = df.loc[condition_date, :]

        df = df.drop(columns=[
            'calendar_year',
            'period',
        ])

        return df

    def get_calendar(self):

        df = pd.DataFrame(self.date_index, columns=['date'])

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

        levels = ['date', ]
        df.set_index(levels, inplace=True)
        df.sort_index(level=levels, ascending=True, inplace=True)

        return df
    

class FinformerDataset(Dataset):

    def __init__(self, data, config):

        self.data = data
        self.config = config

        self.start_date = pd.to_datetime(self.config.start_date, format='%Y-%m-%d').date()
        self.end_date = pd.to_datetime(self.config.end_date, format='%Y-%m-%d').date()

        self.batch_length = self.config.context_length + self.config.prediction_length

        delta = pd.to_timedelta(self.batch_length - 1, unit='D')
        batch_end = self.start_date + delta

        self.batch_date_index = pd.date_range(start=self.start_date, end=batch_end, freq='D')

        self.index_map = self.get_index_map()

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ticker, date_offset = self.index_map[idx]

        ticker_index = ticker
        date_index = self.batch_date_index + pd.to_timedelta(date_offset, unit='D')

        batch_text, batch_num = self.get_batch(ticker_index, date_index)

        return batch_text, batch_num

    def get_index_map(self):

        ticker_index = self.data.tickers

        n_dates = (self.end_date - self.start_date).days + 1
        date_offset_index = list(range(0, n_dates, self.batch_length))

        index_map = dict()

        index = product(ticker_index, date_offset_index)
        for i, (ticker, date_offset) in enumerate(index):
            index_map[i] = (ticker, date_offset)

        return index_map

    def get_batch(self, ticker_index, date_index):

        text, text_pair, text_date = self.get_text(ticker_index, date_index)

        # TODO: Separate batches in dataloader / collate / training to utilize GPU
        batch_values = self.get_batch_values(ticker_index, date_index)
        past_values = batch_values[:, :self.config.context_length, :]
        future_values = batch_values[:, self.config.context_length:, :]

        batch_time_features = self.get_batch_time_features(ticker_index, date_index)
        past_time_features = batch_time_features[:, :self.config.context_length, :]
        future_time_features = batch_time_features[:, self.config.context_length:, :]

        static_categorical_features = self.get_static_categorical_features(ticker_index)
        static_real_features = self.get_static_real_features(ticker_index)

        batch_text = dict(
            text=text,
            text_pair=text_pair,
            text_date=text_date,
        )

        batch_num = dict(
            past_values=past_values,
            past_time_features=past_time_features,
            future_values=future_values,
            future_time_features=future_time_features,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
        )

        return batch_text, batch_num

    def get_text(self, ticker_index, date_index):

        batch_start = date_index.min().date()
        batch_end = date_index.max().date()
        columns = ['title', 'text']
        df_text = self.data.news.loc[pd.IndexSlice[ticker_index, batch_start:batch_end], columns]

        text = df_text['title'].tolist()
        text_pair = df_text['text'].tolist()
        text_date = df_text.index.get_level_values('timestamp').floor(freq='D')

        return text, text_pair, text_date

    def get_batch_values(self, ticker_index, date_index):

        columns = ['open', 'close']
        df_batch_values = self.data.prices.loc[pd.IndexSlice[ticker_index, date_index], columns]

        # [B, C + P, N]
        batch_values = torch.tensor(df_batch_values.values, dtype=torch.float64).unsqueeze(0)

        return batch_values

    def get_batch_observed_mask(self, batch_values):

        batch_observed_mask = batch_values.isnan()

        return batch_observed_mask

    def get_batch_time_features(self, ticker_index, date_index):

        _batch_time_features = self._get_batch_time_features(date_index)
        _batch_dynamic_real_features = self._get_batch_dynamic_real_features(ticker_index, date_index)

        # [B, C + P, N]
        batch_time_features = torch.concat([_batch_time_features, _batch_dynamic_real_features], dim=2)

        return batch_time_features

    def _get_batch_time_features(self, date_index):

        columns = [
            'days_in_month',
            'is_month_end',
            'is_quarter_end',
            'age',
            'weekday_Monday',
            'weekday_Saturday',
            'weekday_Sunday',
            'weekday_Thursday',
            'weekday_Tuesday',
            'weekday_Wednesday',
        ]

        df_batch_time_features = self.data.calendar.loc[date_index, columns]

        # [B, C + P, N]
        batch_time_features = torch.tensor(df_batch_time_features.values.astype('float'), dtype=torch.float64).unsqueeze(0)

        return batch_time_features

    def _get_batch_dynamic_real_features(self, ticker_index, date_index):

        df_batch_dynamic_real_features = self.data.metrics.loc[pd.IndexSlice[ticker_index, :], :]

        batch_start = date_index.min().date()
        batch_end = date_index.max().date()

        condition = (df_batch_dynamic_real_features['start_date'] <= batch_end) & (df_batch_dynamic_real_features['end_date'] >= batch_start)
        columns = [
            'revenue_per_share', 'net_income_per_share', 'market_cap', 'pe_ratio',
            'price_to_sales_ratio', 'pocfratio', 'pfcf_ratio', 'pb_ratio',
            'ptb_ratio', 'debt_to_equity', 'debt_to_assets', 'current_ratio',
            'interest_coverage', 'income_quality',
            'sales_general_and_administrative_to_revenue',
            'research_and_ddevelopement_to_revenue', 'intangibles_to_total_assets',
            'capex_to_operating_cash_flow', 'capex_to_depreciation',
            'invested_capital',
        ]

        df_batch_dynamic_real_features = df_batch_dynamic_real_features.loc[condition, columns]

        df_batch_dynamic_real_features = (
            df_batch_dynamic_real_features
            .reset_index(level=['ticker'], drop=True)
            .reindex(date_index, method='ffill')
        )

        # [B, C + P, N]
        batch_dynamic_real_features = torch.tensor(df_batch_dynamic_real_features.values, dtype=torch.float64).unsqueeze(0)

        return batch_dynamic_real_features

    def get_static_categorical_features(self, ticker_index):

        columns = ['sector_id', 'industry_id', 'country_id']
        df_static_categorical_features = self.data.profile.loc[ticker_index, columns]

        # [B, N]
        static_categorical_features = torch.tensor(df_static_categorical_features.values.astype('int'), dtype=torch.int64).unsqueeze(0)

        return static_categorical_features

    def get_static_real_features(self, ticker_index):

        columns = ['age_ipo']
        df_static_real_features = self.data.profile.loc[ticker_index, columns]

        # [B, N]
        static_real_features = torch.tensor(df_static_real_features.values.astype('float'), dtype=torch.float64).unsqueeze(0)

        return static_real_features


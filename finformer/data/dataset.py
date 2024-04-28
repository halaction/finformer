import os
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import re

from huggingface_hub import login, hf_hub_download
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.preprocessing import LabelEncoder


load_dotenv()
HF_TOKEN = os.environ['HF_TOKEN']

login(token=HF_TOKEN)

DATA_DIR = './data'
SOURCE_DIR = './finformer/data'


def snake_case(string: str):
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    string = pattern.sub('_', string).lower()
    return string


class FinformerConfig:
    repo_id = 'halaction/finformer-data'
    context_length = 30
    prediction_length = 15
    start_date = '2020-01-01'
    end_date = '2024-04-15'


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

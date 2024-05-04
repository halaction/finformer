import os
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from itertools import product

from huggingface_hub import login, hf_hub_download
from sklearn.preprocessing import LabelEncoder


class FinformerData:

    def __init__(self, config, hf_token=None, force=False):

        self.config = config

        self.start_date = pd.to_datetime(self.config.params.start_date, format='%Y-%m-%d').date()
        self.end_date = pd.to_datetime(self.config.params.end_date, format='%Y-%m-%d').date()
        self.date_index = pd.date_range(start=self.start_date, end=self.end_date, freq='D')

        self.sequence_length = self.config.params.context_length + self.config.params.max_lag
        self.prediction_length = self.config.params.prediction_length
        self.batch_length = self.sequence_length + self.prediction_length

        self.keys = ['tickers', 'changes', 'profile', 'metrics', 'prices', 'news', 'calendar']

        os.makedirs(self.config.dirs.dataset_dir, exist_ok=True)

        # self.features = self._get_features()

        hf_token = os.environ['HF_TOKEN']
        login(token=hf_token)

        self.load(force=force)
        self.save(force=force)

        self.label_encoders = self._get_label_encoders()

        # NOTE: Used to avoid copy in child datasets
        self._index = self._get_index()

    def _get_index(self):

        ticker_index = self.tickers
        
        n_dates = (self.end_date - self.start_date).days - self.batch_length + 1 
        date_offset_index = list(range(0, n_dates, self.batch_length))

        index = list(product(ticker_index, date_offset_index))

        return index

    def load(self, force=False):

        progress_bar = tqdm(self.keys)
        for key in progress_bar:
            progress_bar.set_description(desc=f'LOADING (key={key})')

            path = os.path.join(self.config.dirs.dataset_dir, f'{key}.pkl')
            exists = os.path.exists(path)

            if exists and not force:
                df = pd.read_pickle(path)
            else:
                df = self._get_data(key)

            setattr(self, key, df)

        self.filter_data()

    def filter_data(self):

        news_count = self.news['title'].groupby(level='ticker').count()
        prices_count = self.prices['close'].groupby(level='ticker').count()

        index_news = news_count[news_count >= self.config.params.min_news].index
        index_prices = prices_count[prices_count >= self.config.params.min_trading_days].index

        valid_tickers = index_news.intersection(index_prices)

        self.tickers = pd.Series(valid_tickers)
        self.profile = self.profile.loc[self.tickers, :]
        self.metrics = self.metrics.loc[pd.IndexSlice[valid_tickers, :], :]
        self.prices = self.prices.loc[pd.IndexSlice[valid_tickers, :], :]
        self.news = self.news.loc[pd.IndexSlice[valid_tickers, :], :]

        #for key in self.keys:
        #    df = getattr(self, key)
        #    if 'ticker' in df.index.names:

    def save(self, force=False):

        progress_bar = tqdm(self.keys)
        for key in progress_bar:
            progress_bar.set_description(desc=f'SAVING (key={key})')

            path = os.path.join(self.config.dirs.dataset_dir, f'{key}.pkl')
            exists = os.path.exists(path)

            if not exists or force:
                df = getattr(self, key)
                df.to_pickle(path)

    def _get_features(self):
        
        features = dict()

        # Profile
        features['static_categorical_features'] = ['symbol', 'industry', 'sector', 'country']
        features['static_real_features'] = ['age_ipo', ]

        # Metrics
        features['dynamic_real_features'] = [
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

        # News
        features['value_features'] = [
            'open', 
            'close', 
            'low', 
            'high', 
            'volume'
        ]

        # Calendar
        features['time_features'] = [
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
            'month_August',
            'month_December',
            'month_February',
            'month_January',
            'month_July',
            'month_June',
            'month_March',
            'month_May',
            'month_November',
            'month_October',
            'month_September',
            'quarter_2',
            'quarter_3',
            'quarter_4',
        ]
        
        return features

    def _get_label_encoders(self):

        columns = self.config.features.static_categorical_features
        label_encoders = {column: LabelEncoder() for column in columns}

        for column, encoder in label_encoders.items():
            self.profile[column] = encoder.fit_transform(self.profile[column])

        self.config.time_series_model.model.config.cardinality = [
            len(label_encoders[column].classes_) 
            for column in columns
        ]

        return label_encoders

    def _load_csv(self, key):

        filename = os.path.join(self.config.hf.dataset_dir, f'{key}.csv')
        path = hf_hub_download(repo_id=self.config.hf.repo_id, filename=filename, repo_type='dataset')
        df = pd.read_csv(path)

        return df
    
    def _get_data(self, key):

        if key == 'tickers':
            df = self.get_tickers()
        elif key == 'changes':
            df = self.get_changes()
        elif key == 'profile':
            df = self.get_profile()
        elif key == 'metrics':
            df = self.get_metrics()
        elif key == 'prices':
            df = self.get_prices()
        elif key == 'news':
            df = self.get_news()
        elif key == 'calendar':
            df = self.get_calendar()
        else:
            raise ValueError(f'Unknown key `{key}` is provided.')

        return df

    def get_tickers(self):

        key = 'tickers'
        df = self._load_csv(key)

        tickers = pd.Series(df['symbol'].unique())

        return tickers
    
    def get_changes(self):
        
        key = 'changes'
        df = self._load_csv(key)

        changes = df.loc[:, ['oldSymbol', 'newSymbol']]

        return changes
    
    def get_profile(self):

        key = 'profile'
        df = self._load_csv(key)

        df.drop_duplicates(inplace=True)

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

        df['symbol'] = df['ticker']

        levels = ['ticker', ]
        df.set_index(levels, inplace=True)
        df.sort_index(level=levels, ascending=True, inplace=True)

        return df

    def get_metrics(self):

        key = 'metrics'
        df = self._load_csv(key)

        df.drop_duplicates(subset=['symbol', 'date'], inplace=True)

        condition = df['symbol'].isin(self.tickers)

        index = ['symbol', 'date']
        features = self.config.features.dynamic_real_features

        columns = index + features

        df = df.loc[condition, columns]

        df = df.rename(columns={
            'symbol': 'ticker',
        })

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date
        df['start_date'] = df['date']

        levels = ['ticker', 'date']
        df.set_index(levels, inplace=True)
        df.sort_index(level=levels, ascending=True, inplace=True)

        df['end_date'] = df.groupby(level='ticker')['start_date'].shift(periods=-1, fill_value=self.end_date)
        df['end_date'] = df['end_date'] - pd.to_timedelta(1, unit='D')

        condition_date = (df['end_date'] >= self.start_date) & (df['start_date'] <= self.end_date)
        df = df.loc[condition_date, :]

        return df

    def get_news(self):

        key = 'news'
        df = self._load_csv(key)

        df.drop_duplicates(inplace=True)

        condition = df['symbol'].isin(self.tickers)
        columns = ['symbol', 'publishedDate', 'title', 'text']

        df = df.loc[condition, columns]

        df = df.rename(columns={
            'symbol': 'ticker',
            'publishedDate': 'timestamp',
        })

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        df['date'] = df['timestamp'].dt.floor(freq='D')

        condition_date = (df['date'].dt.date >= self.start_date) & (df['date'].dt.date <= self.end_date)
        df = df.loc[condition_date, :]

        level = ['ticker', 'timestamp']
        df.set_index(level, inplace=True)
        df.sort_index(level=level, ascending=True, inplace=True)

        df[['title', 'text']] = df[['title', 'text']].fillna('')

        return df

    def get_prices(self):

        key = 'prices'
        df = self._load_csv(key)

        df.drop_duplicates(subset=['symbol', 'date'], inplace=True)

        condition = df['symbol'].isin(self.tickers)
        index = ['symbol', 'date']
        features = self.config.features.value_features

        columns = index + features

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

        df['_close'] = df.groupby(level='ticker')['close'].ffill().bfill()

        flow_features = ['change', 'changePercent', 'volume']
        stock_features = ['close', 'open', 'low', 'high', 'wvap']

        for feature in flow_features:
            if feature in features:
                df.loc[:, feature].fillna(value=0, inplace=True)

        for feature in stock_features:
            if feature in features:
                df.loc[:, feature].fillna(df['_close'], inplace=True)

        df.drop(columns=['_close', ], inplace=True)

        # Transform target
        if self.config.params.target_transform is None:
            pass
        elif self.config.params.target_transform == 'log':
            df = np.log1p(df)
        else:
            raise ValueError('Unknown target transform.')
        
        return df

    def get_calendar(self):

        df = pd.DataFrame(self.date_index, columns=['date'])

        df['weekday'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month_name()
        df['days_in_month'] = df['date'].dt.days_in_month
        df['is_month_end'] = df['date'].dt.is_month_end
        df['quarter'] = df['date'].dt.quarter
        df['is_quarter_end'] = df['date'].dt.is_quarter_end
        start_year = pd.to_datetime(self.config.params.start_date, format='%Y-%m-%d').year
        df['age'] = df['date'].dt.year - start_year

        columns = ['weekday', 'month', 'quarter']
        df = pd.get_dummies(df, columns=columns, drop_first=True)

        levels = ['date', ]
        df.set_index(levels, inplace=True)
        df.sort_index(level=levels, ascending=True, inplace=True)

        features = df.columns.tolist()
        self.config.features.time_features = features

        return df
    
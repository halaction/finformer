import pandas as pd
from itertools import product, chain
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from finformer.utils import FinformerConfig, filter_none


config = FinformerConfig()

DATA_DIR = config.dirs.data_dir
SOURCE_DIR = config.dirs.source_dir


def get_dataloader(config, dataset):

    collate_fn = FinformerCollator(config)

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=config.params.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
    )

    return dataloader


class FinformerBatch:

    def __init__(
        self, 
        batch_text: Dict,
        batch_num: Dict,
        tickers: List,
        date_offsets: List,
        date_ids: List,
        lengths: List,
    ):
        
        self.batch_text = batch_text
        self.batch_num = batch_num

        self.tickers = tickers
        self.date_offsets = date_offsets
        self.date_ids = date_ids
        self.lengths = lengths


class FinformerCollator:

    def __init__(self, config):
        self.config = config

    @staticmethod
    def right_zero_pad(tensor, max_length, dim=-1):
        
        if tensor.size(dim) < max_length:
            pad = [(0, 0) for _ in range(tensor.dim())]
            pad[dim] = (0, max_length - tensor.size(dim))
            pad = tuple(chain.from_iterable(pad[::-1]))

            padded_tensor = F.pad(tensor, pad, 'constant', 0)
        else:
            padded_tensor = tensor

        return padded_tensor

    def _collate_batch_text(self, batch_text, date_ids):

        keys = None

        # Convert array of dicts to dict of arrays
        for batch_item in batch_text:
            if batch_item is not None:
                if keys is None:
                    keys = batch_item.keys()
                    output = {key: list() for key in keys}

                for key in keys:
                    value = batch_item[key]
                    output[key].append(value)

                # TODO: Avoid copies and delete input batch

        output_values = list()

        # Pad and concatenate tensors inside dict
        for key, values in output.items():

            lengths = [value.size(1) for value in values]
            max_length = max(lengths)
                
            values = [self.right_zero_pad(value, max_length, dim=1) for value in values]

            values_split = torch.cat(values, dim=0).split(self.config.sentiment_model.max_batch_size, dim=0)

            output_values.append(values_split)

        batch_text = list(map(
            lambda output_split: dict(zip(keys, output_split)), 
            zip(*output_values)
        ))

        date_ids = filter_none(date_ids)
        date_ids = torch.cat(date_ids, dim=0).split(self.config.sentiment_model.max_batch_size, dim=0)

        return batch_text, date_ids
    
    def _collate_batch_num(self, batch_num):

        keys = None

        # Convert array of dicts to dict of arrays
        for batch_item in batch_num:
            if batch_item is not None:
                if keys is None:
                    keys = batch_item.keys()
                    output = {key: list() for key in keys}

                for key in keys:
                    value = batch_item[key]
                    output[key].append(value)

        # Concatenate tensors in dict
        for key, values in output.items():
            output[key] = torch.cat(values, dim=0)

        return output

    def collate_fn(self, batch):

        batch_text, batch_num, tickers, date_offsets, date_ids, lengths = zip(*batch)

        batch_text, date_ids = self._collate_batch_text(batch_text, date_ids)

        batch_num = self._collate_batch_num(batch_num)

        collated_batch = FinformerBatch(
            batch_text=batch_text,
            batch_num=batch_num,
            tickers=tickers,
            date_offsets=date_offsets,
            date_ids=date_ids,
            lengths=lengths,
        )

        return collated_batch

    def __call__(self, batch):
        return self.collate_fn(batch)


class FinformerDataset(Dataset):

    def __init__(self, data, config):

        self.data = data
        self.config = config

        self.start_date = pd.to_datetime(self.config.params.start_date, format='%Y-%m-%d').date()
        self.end_date = pd.to_datetime(self.config.params.end_date, format='%Y-%m-%d').date()
        
        self.sequence_length = self.config.params.context_length + self.config.params.max_lag
        self.prediction_length = self.config.params.prediction_length
        self.batch_length = self.sequence_length + self.prediction_length

        delta = pd.to_timedelta(self.batch_length - 1, unit='D')
        batch_end = self.start_date + delta

        self.batch_date_index = pd.date_range(start=self.start_date, end=batch_end, freq='D')

        self.timestamp_freq = int(24 * 60 * 60 * 1e9)
        self.start_date_int = self.batch_date_index.values.astype(int).min() // self.timestamp_freq

        self._index = self._get_index()

        self.tokenizer = self._get_tokenizer()

    def _get_tokenizer(self):

        tokenizer = AutoTokenizer.from_pretrained(self.config.sentiment_model.pretrained_model_name)

        return tokenizer

    def _get_index(self):

        ticker_index = self.data.tickers
        
        n_dates = (self.end_date - self.start_date).days - self.batch_length + 1 
        date_offset_index = list(range(0, n_dates, self.batch_length))

        _index = list(product(ticker_index, date_offset_index))

        return _index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ticker, date_offset = self._index[idx]

        ticker_index = ticker
        date_index = self.batch_date_index + pd.to_timedelta(date_offset, unit='D')

        batch_text, batch_num, date_ids, length = self.get_batch(ticker_index, date_index)

        if date_ids is not None:
            date_ids -= date_offset

        return batch_text, batch_num, ticker, date_offset, date_ids, length

    def get_batch(self, ticker_index, date_index):

        batch_text, date_ids, length = self.get_text(ticker_index, date_index)
        batch_num = self.get_num(ticker_index, date_index)

        for key, value in batch_num.items():
            batch_num[key] = value.unsqueeze(0)

        return batch_text, batch_num, date_ids, length

    def get_text(self, ticker_index, date_index):

        batch_start = date_index.min().date()
        batch_end = date_index.max().date()

        df_text = self.data.news.loc[pd.IndexSlice[ticker_index, batch_start:batch_end], :].reset_index(drop=True)
        
        # For each day sample news not more than specified amount
        # Source: https://stackoverflow.com/a/67871511
        df_text = df_text.sample(frac=1).groupby(by='date').head(self.config.params.max_news_daily)

        text = df_text['title'].tolist()
        text_pair = df_text['text'].tolist()

        length = len(text)

        if length == 0:
            batch_encoding = None
            date_ids = None
        else:
            batch_encoding = self.tokenizer(
                text=text,
                text_pair=text_pair,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=self.config.sentiment_model.max_length,
                return_tensors='pt',
            )

            date_ids = df_text['date'].values.astype(int)
            date_ids = date_ids // self.timestamp_freq - self.start_date_int
            date_ids = torch.tensor(date_ids, dtype=torch.int64)
        
        return batch_encoding, date_ids, length
    
    def get_num(self, ticker_index, date_index):

        # TODO: Separate batches in dataloader / collate / training to utilize GPU
        batch_values = self.get_batch_values(ticker_index, date_index)
        batch_time_features = self.get_batch_time_features(ticker_index, date_index)
        static_categorical_features = self.get_static_categorical_features(ticker_index)
        static_real_features = self.get_static_real_features(ticker_index)

        batch_num = dict(
            batch_values=batch_values,
            batch_time_features=batch_time_features,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
        )

        return batch_num

    def get_batch_values(self, ticker_index, date_index):

        columns = self.config.features.value_features
        df_batch_values = self.data.prices.loc[pd.IndexSlice[ticker_index, date_index], columns]

        # [C + P, N]
        batch_values = torch.tensor(df_batch_values.values, dtype=torch.float64)

        return batch_values

    def get_batch_observed_mask(self, batch_values):

        batch_observed_mask = batch_values.isnan()

        return batch_observed_mask

    def get_batch_time_features(self, ticker_index, date_index):

        _batch_time_features = self._get_batch_time_features(date_index)
        _batch_dynamic_real_features = self._get_batch_dynamic_real_features(ticker_index, date_index)

        # [C + P, N]
        batch_time_features = torch.concat([_batch_time_features, _batch_dynamic_real_features], dim=1)

        return batch_time_features

    def _get_batch_time_features(self, date_index):

        columns = self.config.features.time_features

        df_batch_time_features = self.data.calendar.loc[date_index, columns]

        # [C + P, N]
        batch_time_features = torch.tensor(df_batch_time_features.values.astype('float'), dtype=torch.float64)

        return batch_time_features

    def _get_batch_dynamic_real_features(self, ticker_index, date_index):

        df_batch_dynamic_real_features = self.data.metrics.loc[pd.IndexSlice[ticker_index, :], :]

        batch_start = date_index.min().date()
        batch_end = date_index.max().date()

        condition = (df_batch_dynamic_real_features['start_date'] <= batch_end) & (df_batch_dynamic_real_features['end_date'] >= batch_start)
        columns = self.config.features.dynamic_real_features

        df_batch_dynamic_real_features = df_batch_dynamic_real_features.loc[condition, columns]

        df_batch_dynamic_real_features = (
            df_batch_dynamic_real_features
            .reset_index(level=['ticker'], drop=True)
            .reindex(date_index, method='ffill')
        )

        # [C + P, N]
        batch_dynamic_real_features = torch.tensor(df_batch_dynamic_real_features.values, dtype=torch.float64)

        return batch_dynamic_real_features

    def get_static_categorical_features(self, ticker_index):

        columns = self.config.features.static_categorical_features
        df_static_categorical_features = self.data.profile.loc[ticker_index, columns]

        # [N, ]
        static_categorical_features = torch.tensor(df_static_categorical_features.values.astype('int'), dtype=torch.int64)

        return static_categorical_features

    def get_static_real_features(self, ticker_index):

        columns = self.config.features.static_real_features
        df_static_real_features = self.data.profile.loc[ticker_index, columns]

        # [N, ]
        static_real_features = torch.tensor(df_static_real_features.values.astype('float'), dtype=torch.float64)

        return static_real_features
    

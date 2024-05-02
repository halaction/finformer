import torch
import torch.nn as nn

from transformers import AutoModelForSequenceClassification
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

from finformer.utils import FinformerBatch


class SentimentModel(nn.Module):

    def __init__(self, config):

        super().__init__()
        
        self.config = config
        self.model = self.init_model(config)

        self.batch_size = self.config.params.batch_size
        self.sequence_length = self.config.params.context_length + self.config.params.max_lag
        self.prediction_length = self.config.params.prediction_length
        self.window_length = self.sequence_length + self.prediction_length
        self.output_size = config.sentiment_model.output_size

    def init_model(self, config):
        
        model = AutoModelForSequenceClassification.from_pretrained(config.sentiment_model.model.name)

        if config.sentiment_model.output_type == 'logits':
            pass
        elif config.sentiment_model.output_type == 'features':
            model.classifier = nn.Linear(model.config.hidden_size, config.sentiment_model.output_size)
        else:
            raise ValueError(f'Unknown output_type `{config.sentiment_model.output_type}`.')

        return model

    def forward(self, batch):

        batch_text_splits = batch.pop('batch_text_splits')
        date_ids_splits = batch.pop('date_ids_splits')

        #batch_sentiment = self.batch_output
        #batch_sentiment.fill_(0)

        batch_sentiment = torch.zeros(size=(self.batch_size * self.window_length, self.output_size))

        if len(batch_text_splits) > 0: 
            
            # TODO: Can this loop be automatized? 
            for batch_text_split, date_ids_split in zip(batch_text_splits, date_ids_splits):

                with torch.no_grad():
                    # [B, L] -> [B, D]
                    sentiment_output_split = self.model(**batch_text_split).logits
                
                sentiment_output_split = sentiment_output_split
                batch_sentiment.index_add_(dim=0, index=date_ids_split, source=sentiment_output_split)
            
        batch_sentiment = batch_sentiment.view(self.batch_size, self.window_length, self.output_size)

        # TODO: Make up something better. 
        # > Why would you pass prediction part to model at all then?
        
        # Future mask for news
        batch_sentiment[:, self.sequence_length:, :].fill_(float('nan'))

        return batch_sentiment


class TimeSeriesModel(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config
        self.model = self.init_model(config)

        self.batch_size = self.config.params.batch_size
        self.sequence_length = self.config.params.context_length + self.config.params.max_lag
        self.prediction_length = self.config.params.prediction_length
        self.window_length = self.sequence_length + self.prediction_length
        self.output_size = config.sentiment_model.output_size

    def init_model(self, config):

        input_size = len(config.features.value_features) + config.sentiment_model.output_size
        lags_sequence = list(range(1, self.config.params.max_lag + 1))

        num_time_features = len(config.features.time_features)
        num_dynamic_real_features = len(config.features.dynamic_real_features)

        num_static_categorical_features = len(config.features.static_categorical_features)
        num_static_real_features = len(config.features.static_real_features)

        cardinality = config.time_series_model.cardinality
        embedding_dimension = [config.time_series_model.embedding_dimension for _ in range(num_static_categorical_features)]

        model_config = TimeSeriesTransformerConfig(
            context_length=config.params.context_length,
            prediction_length=config.params.prediction_length,
            distribution_output='student_t',
            loss='nll',
            scaling=None,
            input_size=input_size,
            lags_sequence=lags_sequence,
            num_time_features=num_time_features,
            num_dynamic_real_features=num_dynamic_real_features,
            num_static_categorical_features=num_static_categorical_features,
            num_static_real_features=num_static_real_features,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_head=2,
            encoder_ffn_dim=2,
            decoder_ffn_dim=2,
            activation_function='gelu',
            dropout=0.1,
            encoder_layerdrop=0.1,
            decoder_layerdrop=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,    
            num_parallel_samples=100,
            init_std=2e-2,
            use_cache=True,
        )

        model = TimeSeriesTransformerForPrediction(model_config)

        return model
    
    def _prepare_batch(self, batch):

        batch_num = batch.pop('batch_num')
        batch_sentiment = batch.pop('batch_sentiment')

        batch_values = batch_num.pop('batch_values')
        batch_values = torch.cat([batch_values, batch_sentiment], dim=2)

        observed_mask = batch_values.isnan()
        batch_values.nan_to_num_()

        batch_num.past_values = batch_values[:, :self.sequence_length, :]
        batch_num.future_values = batch_values[:, self.sequence_length:, :]

        batch_num.past_observed_mask = observed_mask[:, :self.sequence_length, :]
        batch_num.future_observed_mask = observed_mask[:, self.sequence_length:, :]

        batch_time_features = batch_num.pop('batch_time_features')
        batch_num.past_time_features = batch_time_features[:, :self.sequence_length, :]
        batch_num.future_time_features = batch_time_features[:, self.sequence_length:, :]

        return batch_num
    
    
    def forward(self, batch):

        batch = self._prepare_batch(batch)
        batch_output = self.model(**batch)

        return batch_output
    
    def generate(self, batch):

        batch = self._prepare_batch(batch)
        
        batch_output = self.model.generate(
            past_values=batch["past_values"],
            past_time_features=batch["past_time_features"],
            past_observed_mask=batch["past_observed_mask"],
            static_categorical_features=batch["static_categorical_features"],
            static_real_features=batch["static_real_features"],
            future_time_features=batch["future_time_features"],
        )
        
        return batch_output
    

class GenerationConfig:
    _from_model_config = False


class FinformerModel(nn.Module):

    def __init__(
        self,
        config
    ):
        super().__init__()
        
        self._config = config

        self.sentiment_model = SentimentModel(config)
        self.time_series_model = TimeSeriesModel(config)

        self.generation_config = GenerationConfig()

    def forward(self, **inputs):

        batch = FinformerBatch(
            **inputs
        )
        
        batch_sentiment = self.sentiment_model(batch)
        batch.batch_sentiment = batch_sentiment

        batch_output = self.time_series_model(batch)

        return batch_output
    
    def generate(self, **inputs):

        batch = FinformerBatch(
            **inputs
        )
        
        batch_sentiment = self.sentiment_model(batch)
        batch.batch_sentiment = batch_sentiment

        batch_output = self.time_series_model.generate(batch)

        print(batch_output)
        print(batch_output.sequences.size())
        print(batch_output.sequences.median(dim=1).values[:4, :, :5])

        return batch_output

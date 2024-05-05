import torch
import torch.nn as nn

from transformers import PreTrainedModel, PretrainedConfig, GenerationConfig
from transformers import AutoModelForSequenceClassification
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from peft import LoraConfig, TaskType, get_peft_model

from hydra.utils import instantiate

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
            model.classifier = nn.Sequential(
                nn.Linear(model.config.hidden_size, 4 * config.sentiment_model.output_size),
                nn.Linear(4 * config.sentiment_model.output_size, config.sentiment_model.output_size),
            )
        else:
            raise ValueError(f'Unknown output_type `{config.sentiment_model.output_type}`.')
        
        if config.sentiment_model.output_type == 'features':
            
            if config.sentiment_model.training_type == 'probing':
                for name, param in model.named_parameters():
                    if not name.startswith('classifier'):
                        param.requires_grad = False

            elif config.sentiment_model.training_type == 'lora':
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS, 
                    inference_mode=False, 
                    r=16, 
                    lora_alpha=32, 
                    lora_dropout=0.1
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            
            else: 
                raise ValueError(f'Unknown training_type `{config.sentiment_model.training_type}`.')

        return model

    def forward(self, batch):

        batch_text_splits = batch.pop('batch_text_splits')
        date_ids_splits = batch.pop('date_ids_splits')

        batch_values = batch.batch_num.batch_values
        batch_size = batch_values.size(0)
        dtype = batch_values.dtype
        device = batch_values.device

        batch_sentiment = torch.zeros(
            size=(batch_size * self.window_length, self.output_size),
            dtype=dtype,
            device=device,
        )

        if self.config.training_args.fp16:
            batch_sentiment = batch_sentiment.half()

        if len(batch_text_splits) > 0: 
            
            # TODO: Can this loop be automatized? 
            for batch_text_split, date_ids_split in zip(batch_text_splits, date_ids_splits):
                # [B, L] -> [B, D]
                sentiment_output_split = self.model(**batch_text_split).logits
                
                batch_sentiment.index_add_(dim=0, index=date_ids_split, source=sentiment_output_split)
            
        batch_sentiment = batch_sentiment.view(batch_size, self.window_length, self.output_size)

        # TODO: Make up something better. 
        # > Why would you pass prediction part to model at all then?
        
        # Future mask for news
        if self.config.params.mask_sentiment:
            batch_sentiment[:, (self.sequence_length + 1):, :].fill_(float('nan'))

        return batch_sentiment


class TimeSeriesModel(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config

        self.batch_size = config.params.batch_size
        self.context_length = config.params.context_length
        self.sequence_length = self.context_length + config.params.max_lag
        self.prediction_length = config.params.prediction_length
        self.window_length = self.sequence_length + self.prediction_length
        self.output_size = config.sentiment_model.output_size

        self.model = self.init_model(config)

    def init_model(self, config):

        model_config = config.time_series_model.model.config

        model_config.prediction_length = self.prediction_length
        model_config.context_length = self.context_length

        model_config.input_size = len(config.features.value_features) + self.output_size
        model_config.lags_sequence = list(range(1, self.config.params.max_lag + 1))

        model_config.num_time_features = len(config.features.time_features)
        model_config.num_dynamic_real_features = len(config.features.dynamic_real_features)

        model_config.num_static_categorical_features = len(config.features.static_categorical_features)
        model_config.num_static_real_features = len(config.features.static_real_features)

        if isinstance(model_config.embedding_dimension, int):
            model_config.embedding_dimension = [model_config.embedding_dimension for _ in range(model_config.num_static_categorical_features)]

        model = instantiate(config.time_series_model.model)

        return model
    
    def _prepare_batch(self, batch):

        batch_num = batch.pop('batch_num')
        batch_sentiment = batch.pop('batch_sentiment')

        batch_values = batch_num.pop('batch_values')
        batch_values = torch.cat([batch_values, batch_sentiment], dim=2)

        observed_mask = ~ batch_values.isnan()
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


class FinformerModel(PreTrainedModel):

    def __init__(
        self,
        config
    ):
        
        pretrained_config = PretrainedConfig(
            name_or_path='finformer-model',
        )

        super().__init__(pretrained_config)
        
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

        return batch_output

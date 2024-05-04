import numpy as np
from copy import deepcopy
from typing import Dict, Union, Any, Optional, List, Tuple

import torch
from torch import nn

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from evaluate import load

from omegaconf import DictConfig

from finformer.config import get_config
from finformer.data.dataset import FinformerCollator, get_split_dataset
from finformer.model import FinformerModel


mape_metric = load('evaluate-metric/mape', 'multilist')
smape_metric = load('evaluate-metric/smape', 'multilist')
mse_metric = load('evaluate-metric/mse', 'multilist')
mae_metric = load('evaluate-metric/mae', 'multilist')

metrics = [mape_metric, smape_metric, mse_metric, mae_metric]


def get_compute_metrics(config):

    value_features = config.features.value_features
    n_value_features = len(value_features)
    target_transform = config.params.target_transform

    def compute_metrics(eval_prediction):

        future_values_pred = eval_prediction.predictions
        future_values = np.nan_to_num(eval_prediction.label_ids)
        
        if target_transform is None:
            pass
        elif target_transform == 'log':
            future_values_pred[:, :, :n_value_features] = np.expm1(future_values_pred)
            future_values = np.expm1(future_values)
        else:
            raise ValueError('Unknown target transform.')

        metrics_values = dict()

        for metric in metrics:
            for i, value_feature in enumerate(value_features):
                name = metric.name

                key = f'{name}/{value_feature}'
                value = metric.compute(
                    predictions=future_values_pred[:, :, i].T, 
                    references=future_values[:, :, i].T
                )[name]

                metrics_values[key] = value

        return metrics_values
    
    return compute_metrics


def get_preprocess_logits_for_metrics(config):

    input_size = len(config.features.value_features)

    def preprocess_logits_for_metrics(logits, labels):

        future_values_pred = logits.sequences.median(dim=1).values[:, :, :input_size]

        return future_values_pred
    
    return preprocess_logits_for_metrics


class FinformerSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(
        self, 
        train_dataset,
        eval_dataset,
        config: DictConfig = None
    ):
        
        if config is None:
            config = get_config()
        
        config.training_args.per_device_train_batch_size = config.params.batch_size
        config.training_args.per_device_eval_batch_size = config.params.batch_size * 4

        self.sequence_length = config.params.context_length + config.params.max_lag
        self._config = config

        data_collator = FinformerCollator(config)

        model = FinformerModel(config)

        training_args = Seq2SeqTrainingArguments(
            **config.training_args,
            include_inputs_for_metrics=False,
            predict_with_generate=True,
            generation_max_length=config.params.prediction_length,
            generation_num_beams=1,
        )

        compute_metrics = get_compute_metrics(config)
        preprocess_logits_for_metrics = get_preprocess_logits_for_metrics(config)

        # Init Trainer
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_future_values = 'batch_values' in inputs['batch_num']
        inputs = self._prepare_inputs(inputs)

        if len(gen_kwargs) == 0 and hasattr(self, '_gen_kwargs'):
            gen_kwargs = self._gen_kwargs.copy()
        if 'num_beams' in gen_kwargs and gen_kwargs['num_beams'] is None:
            gen_kwargs.pop('num_beams')
        if 'max_length' in gen_kwargs and gen_kwargs['max_length'] is None:
            gen_kwargs.pop('max_length')

        default_synced_gpus = False
        gen_kwargs['synced_gpus'] = (
            gen_kwargs['synced_gpus'] if gen_kwargs.get('synced_gpus') is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()

        if (
            'future_values' in generation_inputs
            and 'decoder_input_ids' in generation_inputs
            and generation_inputs['future_values'].shape == generation_inputs['decoder_input_ids'].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ('decoder_input_ids', 'decoder_attention_mask')
            }

        generated_values = self.model.generate(**generation_inputs, **gen_kwargs)

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        with torch.no_grad():
            if has_future_values:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                
                loss = (outputs.loss if isinstance(outputs, dict) else outputs[0]).mean().detach()
                    
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_future_values:
            future_values = inputs['batch_num']['batch_values'][:, self.sequence_length:, :]
        else:
            future_values = None
        
        return loss, generated_values, future_values

        
from copy import deepcopy
from typing import Dict, Union, Any, Optional, List, Tuple

import torch
from torch import nn

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from evaluate import load

from omegaconf import DictConfig

from finformer.config import get_config
from finformer.data.dataset import FinformerCollator, get_split_dataset
from finformer.model import FinformerModel


mase_metric = load("evaluate-metric/mase")
mape_metric = load("evaluate-metric/mape")
smape_metric = load("evaluate-metric/smape")

metrics = [mase_metric, mape_metric, smape_metric]


def get_compute_metrics(config):

    def compute_metrics(eval_prediction):

        print('METRICS')

        print(eval_prediction)

        predictions = eval_prediction.predictions
        label_ids = eval_prediction.label_ids
        # predictions = predictions[:, 0]

        print(predictions.shape)
        print(label_ids.shape)

        metrics_values = {
            metric.name: metric.compute(predictions=predictions, references=label_ids) 
            for metric in metrics
        }

        return metrics_values
    
    return compute_metrics


def get_preprocess_logits_for_metrics(config):

    sequence_length = config.params.context_length + config.params.max_lag
    input_size = len(config.features.value_features)

    def preprocess_logits_for_metrics(logits, labels):

        print('PREPROCESS')

        future_values_pred = logits.sequences.median(dim=1).values[:, :, :input_size]
        future_values = labels[:, sequence_length:, :]

        return future_values_pred, future_values
    
    return preprocess_logits_for_metrics


class FinformerSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(self, config: DictConfig = None):
        
        if config is None:
            config = get_config()
        
        config.training_args.per_device_train_batch_size = config.params.batch_size
        config.training_args.per_device_eval_batch_size = config.params.batch_size * 4

        self._config = config

        dataset_train, dataset_val, dataset_test = get_split_dataset(config)
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
            train_dataset=dataset_train,
            eval_dataset={
                'val': dataset_val, 
                'test': dataset_test
            },
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

        has_labels = "batch_num" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        # default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        default_synced_gpus = False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "future_values" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["future_values"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs['batch_num']['batch_values']
        #    if labels.shape[-1] < gen_config.max_length:
        #        labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
        #    elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
        #        labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None


        return loss, generated_tokens, labels

        
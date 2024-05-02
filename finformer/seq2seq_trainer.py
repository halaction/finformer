from copy import deepcopy

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


def compute_metrics(eval_prediction):

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


class MetricsCallback(TrainerCallback):
    
    def __init__(self, trainer):
        super().__init__()
        self._trainer = trainer

    def _callback(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix='train')
            return control_copy

    def on_log(self, args, state, control, **kwargs):
        return self._callback(args, state, control, **kwargs)
    
    def on_step_end(self, args, state, control, **kwargs):
        return self._callback(args, state, control, **kwargs)

    def on_epoch_end(self, args, state, control, **kwargs):
        return self._callback(args, state, control, **kwargs)


class FinformerSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(self, config: DictConfig = None):
        
        if config is None:
            config = get_config()
        
        config.training_args.per_device_train_batch_size = config.params.batch_size
        config.training_args.per_device_eval_batch_size = config.params.batch_size

        self._config = config

        dataset_train, dataset_val, dataset_test = get_split_dataset(config)
        data_collator = FinformerCollator(config)

        model = FinformerModel(config)

        training_args = Seq2SeqTrainingArguments(
            **config.training_args,
            predict_with_generate=True,
            generation_max_length=config.params.prediction_length,
            generation_num_beams=1,
        )

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
        )

        callback = MetricsCallback(self)
        self.add_callback(callback) 
        
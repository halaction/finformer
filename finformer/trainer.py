from copy import deepcopy

from transformers import Trainer, TrainingArguments, TrainerCallback
from evaluate import load

from finformer.utils import FinformerConfig
from finformer.data.dataset import FinformerCollator, get_split_dataset
from finformer.model import FinformerModel


mase_metric = load("evaluate-metric/mase")
mape_metric = load("evaluate-metric/mape")
smape_metric = load("evaluate-metric/smape")

print(mase_metric)
print(mase_metric.__dir__())

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
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def _callback(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix='train')
            return control_copy

    def on_log(self, args, state, control, **kwargs):
        return self._callback(args, state, control, **kwargs)

    def on_epoch_end(self, args, state, control, **kwargs):
        return self._callback(args, state, control, **kwargs)


class FinformerTrainer(Trainer):

    def __init__(self, config: FinformerConfig = None):
        
        if config is None:
            config = FinformerConfig()

        self._config = config

        dataset_train, dataset_val, dataset_test = get_split_dataset(config)
        data_collator = FinformerCollator(config)

        model = FinformerModel(config)

        training_args = TrainingArguments(**config.training_args)

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
        
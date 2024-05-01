from transformers import Trainer, TrainingArguments
from evaluate import load

from finformer.utils import FinformerConfig
from finformer.data.dataset import FinformerCollator, get_split_dataset
from finformer.model import FinformerModel


mase_metric = load("evaluate-metric/mase")
mape_metric = load("evaluate-metric/mape")
smape_metric = load("evaluate-metric/smape")

print(mase_metric)
print(mase_metric.__dir__())

metrics = {
    'mase': mase_metric, 
    'mape': mape_metric, 
    'smape': smape_metric,
}


def compute_metrics(eval_prediction):

    predictions = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    # predictions = predictions[:, 0]

    metrics_values = {
        key: metric.compute(predictions=predictions, references=label_ids) 
        for key, metric in metrics.items()
    }

    return metrics_values


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
            eval_dataset=dataset_val,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        
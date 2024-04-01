from finformer.data.alphavantage import get_data, get_dataloaders
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


def get_model():

    config = TimeSeriesTransformerConfig(
        prediction_length=prediction_length,
        # context length:
        context_length=prediction_length * 2,
        # lags coming from helper given the freq:
        lags_sequence=lags_sequence,
        # we'll add 2 time features ("month of year" and "age", see further):
        num_time_features=len(time_features) + 1,
        # we have a single static categorical feature, namely time series ID:
        num_static_categorical_features=1,
        # it has 366 possible values:
        cardinality=[len(train_dataset)],
        # the model will learn an embedding of size 2 for each of the 366 possible values:
        embedding_dimension=[2],

        # transformer params:
        encoder_layers=4,
        decoder_layers=4,
        d_model=32,
    )

    model = TimeSeriesTransformerForPrediction(config)

    return model


class SentimentStockPredictionModel(nn.Module):
    def init(self, pretrained_model_name, num_classes=1):
        super(SentimentStockPredictionModel, self).init()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


# Training loop
def train_model(model, train_dataloader, optimizer, criterion, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels.float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')


# Evaluation loop
def evaluate_model(model, test_dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels.float())

            total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    return avg_loss



if __name__ == '__main__':
    train_dataloader, test_dataloader = get_dataloaders()

    # Instantiate the model
    model = SentimentStockPredictionModel('bert-base-uncased')

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    train_model(model, train_dataloader, optimizer, criterion)
    test_loss = evaluate_model(model, test_dataloader, criterion)
    print(f'Test Loss: {test_loss:.4f}')

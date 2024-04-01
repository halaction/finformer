import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange
from torchsummary import summary
import os

from finformer.data.alphavantage import get_data, get_dataloaders

class Trainer:

    def __init__(
        self,
        model_name: str = 'resnet101',
        batch_size: int = 128,
        learning_rate: int = 1e-3,
        accumulation_freq: int = None,
        tqdm_freq: int = 10,
        device: str = 'auto',
    ):

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model_name = model_name

        self.batch_size = batch_size

        self.learning_rate = learning_rate

        self.dataloader_train = self._init_dataloader(train=True)
        self.dataloader_test = self._init_dataloader(train=False)

        self.teacher, self.optimizer_teacher = self._init_teacher()
        self.student, self.optimizer_student= self._init_student()

        if model_name == 'resnet18':
            adapter1 = nn.Linear(64, 64, bias=False).to(self.device)
            adapter2 = nn.Linear(128, 128, bias=False).to(self.device)
            adapter4 = nn.Linear(512, 512, bias=False).to(self.device)
            self.adapters = (adapter1, adapter2, adapter4)
        elif model_name == 'resnet101':
            adapter1 = nn.Linear(256, 256, bias=False).to(self.device)
            adapter2 = nn.Linear(512, 512, bias=False).to(self.device)
            adapter4 = nn.Linear(2048, 2048, bias=False).to(self.device)
            self.adapters = (adapter1, adapter2, adapter4)

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        self.max_epochs = 10
        self.max_triggers = 2
        self.tqdm_freq = tqdm_freq
        self.accumulation_freq = accumulation_freq

        self.path_checkpoint = './data'


    def _init_teacher(self) -> nn.Module:

        if self.model_name == 'resnet101':
            teacher = resnet101(weights=ResNet101_Weights.DEFAULT)
            teacher.fc = nn.Linear(in_features=2048, out_features=10)

        elif self.model_name == 'resnet18':
            teacher = resnet18(weights=ResNet18_Weights.DEFAULT)
            teacher.fc = nn.Linear(in_features=512, out_features=10)

        teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False)
        teacher.maxpool = nn.Identity()

        teacher.to(self.device)

        optimizer = torch.optim.Adam(teacher.parameters(), lr=self.learning_rate)

        return teacher, optimizer


    def _init_student(self) -> nn.Module:

        if self.model_name == 'resnet101':
            student = resnet101(weights=None)
            student.layer3 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            student.fc = nn.Linear(in_features=2048, out_features=10)

        elif self.model_name == 'resnet18':
            student = resnet18(weights=None)
            student.layer3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            student.fc = nn.Linear(in_features=512, out_features=10)

        student.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False)
        student.maxpool = nn.Identity()

        student.to(self.device)

        optimizer = torch.optim.Adam(student.parameters(), lr=self.learning_rate)

        return student, optimizer


    def summary(self):

        print('Teacher model:')
        summary(self.teacher, input_size=(3, 32, 32))
        print('\n\n\n')
        print('Student model:')
        summary(self.student, input_size=(3, 32, 32))


    def _init_dataloader(self, train: bool = True) -> DataLoader:

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if not os.path.exists('./data'):
            os.mkdir('./data')

        dataset = get_data(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=1,
        )

        return dataloader


    def save(self, which: str, tag: str):

        if which == 'student':
            model = self.student
        elif which == 'teacher':
            model = self.teacher

        path = os.path.join(self.path_checkpoint, f'{self.model_name}-{which}-{tag}.pt')
        torch.save(model.state_dict(), path)


    def load(self, which: str, tag: str):

        if which == 'student':
            model = self.student
        elif which == 'teacher':
            model = self.teacher

        path = os.path.join(self.path_checkpoint, f'{self.model_name}-{which}-{tag}.pt')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)

        if which == 'student':
            self.optimizer_student = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        elif which == 'teacher':
            self.optimizer_teacher = torch.optim.Adam(model.parameters(), lr=self.learning_rate)


    def _train(self, model, optimizer):

        prev_accuracy = None
        n_triggers = 0

        n_batches_train = len(self.dataloader_train)
        n_batches_test = len(self.dataloader_test)

        for epoch in range(self.max_epochs):

            # Train phase

            running_loss = torch.zeros(1).to(self.device)
            running_accuracy = torch.zeros(1).to(self.device)

            epoch_loss = torch.zeros(1).to(self.device)
            epoch_accuracy = torch.zeros(1).to(self.device)

            progress_bar = tqdm(enumerate(self.dataloader_train), total=n_batches_train)

            model.train()
            for i, batch in progress_bar:
                inputs, labels = batch

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)

                loss = self.ce_loss(outputs, labels)

                if self.accumulation_freq is None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    loss /= self.accumulation_freq
                    loss.backward()

                    if ((i + 1) % self.accumulation_freq == 0) or (i + 1 == n_batches_train):
                        optimizer.step()
                        optimizer.zero_grad()

                accuracy = (outputs.detach().argmax(dim=1) == labels).float().mean()

                running_loss += loss.detach()
                running_accuracy += accuracy.detach()

                if (i + 1) % self.tqdm_freq == 0:
                    mean_loss = running_loss.item() / self.tqdm_freq
                    mean_accuracy = running_accuracy.item() / self.tqdm_freq

                    progress_bar.set_description(f'Train (epoch: {epoch:02} | loss: {mean_loss:.4f} | accuracy: {mean_accuracy:.4f})')

                    epoch_loss += running_loss
                    epoch_accuracy += running_accuracy

                    running_loss[0] = 0.
                    running_accuracy[0] = 0.

            epoch_loss += running_loss
            epoch_accuracy += running_accuracy

            epoch_loss = epoch_loss.item() / n_batches_train
            epoch_accuracy = epoch_accuracy.item() / n_batches_train

            print(f'Train metrics @ Epoch {epoch:02}: \n  - Loss: {epoch_loss:.4f} \n  - Accuracy: {epoch_accuracy:.4f} \n')

            # Test phase

            running_loss = torch.zeros(1).to(self.device)
            running_accuracy = torch.zeros(1).to(self.device)

            epoch_loss = torch.zeros(1).to(self.device)
            epoch_accuracy = torch.zeros(1).to(self.device)

            progress_bar = tqdm(enumerate(self.dataloader_test), total=len(self.dataloader_test))

            model.eval()
            for i, batch in progress_bar:
                inputs, labels = batch

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with torch.inference_mode():
                    outputs = model(inputs)

                    loss = self.ce_loss(outputs, labels)
                    accuracy = (outputs.detach().argmax(dim=1) == labels).float().mean()

                running_loss += loss.detach()
                running_accuracy += accuracy.detach()

                if (i + 1) % self.tqdm_freq == 0:
                    mean_loss = running_loss.item() / self.tqdm_freq
                    mean_accuracy = running_accuracy.item() / self.tqdm_freq

                    progress_bar.set_description(f'Test (epoch: {epoch:02} | loss: {mean_loss:.4f} | accuracy: {mean_accuracy:.4f})')

                    epoch_loss += running_loss
                    epoch_accuracy += running_accuracy

                    running_loss[0] = 0.
                    running_accuracy[0] = 0.

            epoch_loss += running_loss
            epoch_accuracy += running_accuracy

            epoch_loss = epoch_loss.item() / n_batches_test
            epoch_accuracy = epoch_accuracy.item() / n_batches_test

            print(f'Test metrics @ Epoch {epoch:02}: \n  - Loss: {epoch_loss:.4f} \n  - Accuracy: {epoch_accuracy:.4f} \n')

            if prev_accuracy is not None:

                if abs(epoch_accuracy - prev_accuracy) < 0.01:
                    n_triggers += 1
                else:
                    n_triggers = 0

                if n_triggers >= self.max_triggers:
                    print('Stopping criterion was triggered! Training is finished.')
                    break

            prev_accuracy = epoch_accuracy

    def train_teacher(self):
        print('Training teacher model:\n')

        which = 'teacher'
        tag = 'input'

        path = os.path.join(self.path_checkpoint, f'{self.model_name}-{which}-{tag}.pt')

        if os.path.exists(path):
            self.load(which=which, tag=tag)
            print('Model was loaded from cache.')
        else:
            self._train(self.teacher, self.optimizer_teacher)
            self.save(which=which, tag=tag)

    def train_student(self):
        print('Training student model:\n')

        which = 'student'
        tag = 'input'

        path = os.path.join(self.path_checkpoint, f'{self.model_name}-{which}-{tag}.pt')

        if os.path.exists(path):
            self.load(which=which, tag=tag)
            print('Model was loaded from cache.')
        else:
            self._train(self.student, self.optimizer_student)
            self.save(which=which, tag=tag)

    def train_distillation(self):
        print('Training distillation:\n')

        which = 'student'
        tag = 'distillation'

        path = os.path.join(self.path_checkpoint, f'{self.model_name}-{which}-{tag}.pt')

        if os.path.exists(path):
            self.load(which=which, tag=tag)
            print('Model was loaded from cache.')

        else:

            # Ensure student model is untrained
            self.student, self.optimizer_student = self._init_student()

            prev_accuracy = None
            n_triggers = 0

            n_batches_train = len(self.dataloader_train)
            n_batches_test = len(self.dataloader_test)

            self.teacher.eval()
            for epoch in range(self.max_epochs):

                # Train phase

                running_loss = torch.zeros(1).to(self.device)
                running_accuracy = torch.zeros(1).to(self.device)

                epoch_loss = torch.zeros(1).to(self.device)
                epoch_accuracy = torch.zeros(1).to(self.device)

                progress_bar = tqdm(enumerate(self.dataloader_train), total=n_batches_train)

                self.student.train()
                for i, batch in progress_bar:
                    inputs, labels = batch

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.no_grad():
                        outputs_teacher = F.softmax(self.teacher(inputs), dim=-1)

                    outputs_student = self.student(inputs)

                    # Prediction loss
                    loss_student = self.ce_loss(outputs_student, labels)

                    # Soft-label loss
                    loss_distillation = self.ce_loss(outputs_student, outputs_teacher)

                    # Total loss
                    loss = (loss_student + loss_distillation) / 2

                    if self.accumulation_freq is None:
                        self.optimizer_student.zero_grad()
                        loss.backward()
                        self.optimizer_student.step()
                    else:
                        loss /= self.accumulation_freq
                        loss.backward()

                        if ((i + 1) % self.accumulation_freq == 0) or (i + 1 == n_batches_train):
                            self.optimizer_student.step()
                            self.optimizer_student.zero_grad()

                    accuracy = (outputs_student.detach().argmax(dim=1) == labels).float().mean()

                    running_loss += loss.detach()
                    running_accuracy += accuracy.detach()

                    if (i + 1) % self.tqdm_freq == 0:
                        mean_loss = running_loss.item() / self.tqdm_freq
                        mean_accuracy = running_accuracy.item() / self.tqdm_freq

                        progress_bar.set_description(f'Train (epoch: {epoch:02} | loss: {mean_loss:.4f} | accuracy: {mean_accuracy:.4f})')

                        epoch_loss += running_loss
                        epoch_accuracy += running_accuracy

                        running_loss[0] = 0.
                        running_accuracy[0] = 0.

                epoch_loss += running_loss
                epoch_accuracy += running_accuracy

                epoch_loss = epoch_loss.item() / n_batches_train
                epoch_accuracy = epoch_accuracy.item() / n_batches_train

                print(f'Train metrics @ Epoch {epoch:02}: \n  - Loss: {epoch_loss:.4f} \n  - Accuracy: {epoch_accuracy:.4f} \n')

                # Test phase

                running_loss = torch.zeros(1).to(self.device)
                running_accuracy = torch.zeros(1).to(self.device)

                epoch_loss = torch.zeros(1).to(self.device)
                epoch_accuracy = torch.zeros(1).to(self.device)

                progress_bar = tqdm(enumerate(self.dataloader_test), total=len(self.dataloader_test))

                self.student.eval()
                for i, batch in progress_bar:
                    inputs, labels = batch

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.inference_mode():
                        outputs = self.student(inputs)

                        loss = self.ce_loss(outputs, labels)
                        accuracy = (outputs.detach().argmax(dim=1) == labels).float().mean()

                    running_loss += loss.detach()
                    running_accuracy += accuracy.detach()

                    if (i + 1) % self.tqdm_freq == 0:
                        mean_loss = running_loss.item() / self.tqdm_freq
                        mean_accuracy = running_accuracy.item() / self.tqdm_freq

                        progress_bar.set_description(f'Test (epoch: {epoch:02} | loss: {mean_loss:.4f} | accuracy: {mean_accuracy:.4f})')

                        epoch_loss += running_loss
                        epoch_accuracy += running_accuracy

                        running_loss[0] = 0.
                        running_accuracy[0] = 0.

                epoch_loss += running_loss
                epoch_accuracy += running_accuracy

                epoch_loss = epoch_loss.item() / n_batches_test
                epoch_accuracy = epoch_accuracy.item() / n_batches_test

                print(f'Test metrics @ Epoch {epoch:02}: \n  - Loss: {epoch_loss:.4f} \n  - Accuracy: {epoch_accuracy:.4f} \n')

                if prev_accuracy is not None:

                    if abs(epoch_accuracy - prev_accuracy) < 0.01:
                        n_triggers += 1
                    else:
                        n_triggers = 0

                    if n_triggers >= self.max_triggers:
                        print('Stopping criterion was triggered! Training is finished.')
                        break

                prev_accuracy = epoch_accuracy

            self.save(which=which, tag=tag)


    @staticmethod
    def _forward_with_hidden_states(model, x):
        """
        Overwritten static ResNet._forward_impl
        """

        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x1 = x

        x = model.layer2(x)
        x2 = x

        x = model.layer3(x)

        x = model.layer4(x)
        x4 = x

        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(x)

        output = x
        hidden_states = (x1, x2, x4)

        return output, hidden_states


    def train_layer_distillation(self):

        print('Training layer distillation:\n')

        which = 'student'
        tag = 'layer-distillation'

        path = os.path.join(self.path_checkpoint, f'{self.model_name}-{which}-{tag}.pt')

        if os.path.exists(path):
            self.load(which=which, tag=tag)
            print('Model was loaded from cache.')

        else:

            # Ensure student model is untrained
            self.student, self.optimizer_student = self._init_student()

            prev_accuracy = None
            n_triggers = 0

            n_batches_train = len(self.dataloader_train)
            n_batches_test = len(self.dataloader_test)

            self.teacher.eval()

            for epoch in range(self.max_epochs):

                # Train phase

                running_loss = torch.zeros(1).to(self.device)
                running_accuracy = torch.zeros(1).to(self.device)

                epoch_loss = torch.zeros(1).to(self.device)
                epoch_accuracy = torch.zeros(1).to(self.device)

                progress_bar = tqdm(enumerate(self.dataloader_train), total=n_batches_train)

                self.student.train()
                for i, batch in progress_bar:
                    inputs, labels = batch

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.no_grad():
                        outputs_teacher, hidden_states_teacher = self._forward_with_hidden_states(self.teacher, inputs)
                        outputs_teacher = F.softmax(
                            outputs_teacher,
                            dim=-1
                        )

                    outputs_student, hidden_states_student = self._forward_with_hidden_states(self.student, inputs)

                    # Prediction loss
                    loss_student = self.ce_loss(outputs_student, labels)

                    # Soft-label distillation loss
                    loss_distillation = self.ce_loss(outputs_student, outputs_teacher)

                    # Layer distillation loss
                    n_hidden_states = len(self.adapters)
                    _loss_layer_list = []

                    for layer_idx in range(n_hidden_states):
                        adapter = self.adapters[layer_idx]

                        # Move channel to last dim before Linear
                        hidden_state_student = hidden_states_student[layer_idx].permute(0, 2, 3, 1)
                        hidden_state_teacher = hidden_states_teacher[layer_idx].permute(0, 2, 3, 1)

                        _loss_layer = self.mse_loss(adapter(hidden_state_student), hidden_state_teacher)
                        _loss_layer_list.append(_loss_layer)

                    loss_layer = sum(_loss_layer_list) / n_hidden_states

                    # Total loss
                    loss = (loss_student + loss_distillation + loss_layer) / 3

                    if self.accumulation_freq is None:
                        self.optimizer_student.zero_grad()
                        loss.backward()
                        self.optimizer_student.step()
                    else:
                        loss /= self.accumulation_freq
                        loss.backward()

                        if ((i + 1) % self.accumulation_freq == 0) or (i + 1 == n_batches_train):
                            self.optimizer_student.step()
                            self.optimizer_student.zero_grad()

                    accuracy = (outputs_student.detach().argmax(dim=1) == labels).float().mean()

                    running_loss += loss.detach()
                    running_accuracy += accuracy.detach()

                    if (i + 1) % self.tqdm_freq == 0:
                        mean_loss = running_loss.item() / self.tqdm_freq
                        mean_accuracy = running_accuracy.item() / self.tqdm_freq

                        progress_bar.set_description(f'Train (epoch: {epoch:02} | loss: {mean_loss:.4f} | accuracy: {mean_accuracy:.4f})')

                        epoch_loss += running_loss
                        epoch_accuracy += running_accuracy

                        running_loss[0] = 0.
                        running_accuracy[0] = 0.

                epoch_loss += running_loss
                epoch_accuracy += running_accuracy

                epoch_loss = epoch_loss.item() / n_batches_train
                epoch_accuracy = epoch_accuracy.item() / n_batches_train

                print(f'Train metrics @ Epoch {epoch:02}: \n  - Loss: {epoch_loss:.4f} \n  - Accuracy: {epoch_accuracy:.4f} \n')

                # Test phase

                running_loss = torch.zeros(1).to(self.device)
                running_accuracy = torch.zeros(1).to(self.device)

                epoch_loss = torch.zeros(1).to(self.device)
                epoch_accuracy = torch.zeros(1).to(self.device)

                progress_bar = tqdm(enumerate(self.dataloader_test), total=len(self.dataloader_test))

                self.student.eval()
                for i, batch in progress_bar:
                    inputs, labels = batch

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.inference_mode():
                        outputs = self.student(inputs)

                        loss = self.ce_loss(outputs, labels)
                        accuracy = (outputs.detach().argmax(dim=1) == labels).float().mean()

                    running_loss += loss.detach()
                    running_accuracy += accuracy.detach()

                    if (i + 1) % self.tqdm_freq == 0:
                        mean_loss = running_loss.item() / self.tqdm_freq
                        mean_accuracy = running_accuracy.item() / self.tqdm_freq

                        progress_bar.set_description(f'Test (epoch: {epoch:02} | loss: {mean_loss:.4f} | accuracy: {mean_accuracy:.4f})')

                        epoch_loss += running_loss
                        epoch_accuracy += running_accuracy

                        running_loss[0] = 0.
                        running_accuracy[0] = 0.

                epoch_loss += running_loss
                epoch_accuracy += running_accuracy

                epoch_loss = epoch_loss.item() / n_batches_test
                epoch_accuracy = epoch_accuracy.item() / n_batches_test

                print(f'Test metrics @ Epoch {epoch:02}: \n  - Loss: {epoch_loss:.4f} \n  - Accuracy: {epoch_accuracy:.4f} \n')

                if prev_accuracy is not None:

                    if abs(epoch_accuracy - prev_accuracy) < 0.01:
                        n_triggers += 1
                    else:
                        n_triggers = 0

                    if n_triggers >= self.max_triggers:
                        print('Stopping criterion was triggered! Training is finished.')
                        break

                prev_accuracy = epoch_accuracy

            self.save(which=which, tag=tag)

    def evaluate(self, which: str, tag: str):

        self.load(which=which, tag=tag)

        if which == 'teacher':
            model = self.teacher
        elif which == 'student':
            model = self.student

        n_batches_test = len(self.dataloader_test)

        running_loss = torch.zeros(1).to(self.device)
        running_accuracy = torch.zeros(1).to(self.device)

        epoch_loss = torch.zeros(1).to(self.device)
        epoch_accuracy = torch.zeros(1).to(self.device)

        progress_bar = tqdm(enumerate(self.dataloader_test), total=len(self.dataloader_test))

        model.eval()
        for i, batch in progress_bar:
            inputs, labels = batch

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.inference_mode():
                outputs = model(inputs)

                loss = self.ce_loss(outputs, labels)
                accuracy = (outputs.detach().argmax(dim=1) == labels).float().mean()

            running_loss += loss.detach()
            running_accuracy += accuracy.detach()

            if (i + 1) % self.tqdm_freq == 0:
                mean_loss = running_loss.item() / self.tqdm_freq
                mean_accuracy = running_accuracy.item() / self.tqdm_freq

                progress_bar.set_description(f'Evaluation (loss: {mean_loss:.4f} | accuracy: {mean_accuracy:.4f})')

                epoch_loss += running_loss
                epoch_accuracy += running_accuracy

                running_loss[0] = 0.
                running_accuracy[0] = 0.

        epoch_loss += running_loss
        epoch_accuracy += running_accuracy

        epoch_loss = epoch_loss.item() / n_batches_test
        epoch_accuracy = epoch_accuracy.item() / n_batches_test

        print(f'Test metrics for {self.model_name}-{which}-{tag}: \n  - Loss: {epoch_loss:.4f} \n  - Accuracy: {epoch_accuracy:.4f} \n')


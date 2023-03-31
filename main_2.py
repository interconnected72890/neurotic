import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from models import LinearModel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, f1_score
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data, labels, trans=None):
        self.data = data
        self.labels = labels
        self.transform = trans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        if self.transform:
            x = self.transform(x)

        return x, y


class ToTensor1D(object):
    def __call__(self, data):
        return torch.tensor(data, dtype=torch.float32)


for zone in range(1, 4):
    x_train = pd.read_csv("cleaned_tetuan/tetuan_data_zone" + str(zone) + "_x_train.csv").to_numpy()
    y_train = pd.read_csv("cleaned_tetuan/tetuan_data_zone" + str(zone) + "_y_train.csv").to_numpy()
    x_test = pd.read_csv("cleaned_tetuan/tetuan_data_zone" + str(zone) + "_x_test.csv").to_numpy()
    y_test = pd.read_csv("cleaned_tetuan/tetuan_data_zone" + str(zone) + "_y_test.csv").to_numpy()


    # transform = transforms.Compose([transforms.ToTensor()])

    dataset_train = CustomDataset(data=x_train, labels=y_train, trans=ToTensor1D())
    dataset_test = CustomDataset(data=x_test, labels=y_test, trans=ToTensor1D())
    # print(dataset_train.data[:10])
    # print(dataset_train.labels[:10])

    dataloader = DataLoader(dataset_train, batch_size=32, shuffle=True)


    input_size = len(dataset_train.data[0])
    epochs = 50

    for x in range(5):
        model = LinearModel(input_size)
        model.train()

        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(epochs):

            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = torch.tensor(inputs, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.float32)

                outputs = model(inputs)
                loss = loss_func(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print('Epoch [{}/{}], Loss: {}'.format(epoch + 1, epochs, running_loss / len(dataloader)))


        dataloader = DataLoader(dataset_test, batch_size=32, shuffle=False)

        model.eval()

        y_pred = []
        y_true = []

        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data


                outputs = model(inputs)

                y_pred.extend(outputs.squeeze().tolist())
                y_true.extend(labels.squeeze().tolist())

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f'Root Mean Squared Error:  {np.sqrt(mse)}')
        print(f'Mean Squared Error:       {mse}')
        print(f'Mean Absolute Error:      {mae}')
        print(f'R2 Score:                 {r2}')

        with open("acquired_data/tetuan_zone" + str(zone) + "_results.csv", "a") as file:
            file.write(f'{np.sqrt(mse)},{mse},{mae},{r2}\n')
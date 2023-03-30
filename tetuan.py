import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from sklearn.model_selection import train_test_split


class Tetuan:

    def __init__(self):
        self.path = "/Users/hacker/PycharmProjects/UROP/Tetuan City power consumption.csv"
        self.columns = ['Temperature', 'Humidity', 'Wind Speed',
                        'general diffuse flows', 'diffuse flows', 'Zone 1 Power Consumption',
                        'Zone 2  Power Consumption', 'Zone 3  Power Consumption']
        self.data_total = None
        self.data_total_1 = None
        self.data_total_2 = None
        self.data_total_3 = None

        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    def split_data(self, rand_state=7):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train,
                                                                                test_size=0.3, random_state=rand_state)

    def get_data(self):
        if self.data_total is None:
            self.data_total = pd.read_csv(self.path)
            self.data_total = self.data_total.drop(['DateTime'], axis=1)

            """
            Temperature  -  Normalize
            Humidity  -  Normalize
            Wind Speed  -  Normalize
            general diffuse flows  -  Normalize
            diffuse flows  -  Normalize
            
            ****
            Zone 1 Power Consumption  -  Normalize
            Zone 2  Power Consumption  -  Normalize
            Zone 3  Power Consumption  -  Normalize
            ****
            """
            # Temperature
            self.data_total["Temperature"][:] -= self.data_total["Temperature"].min()
            self.data_total["Temperature"][:] /= self.data_total["Temperature"].max()
            # Humidity
            self.data_total["Humidity"][:] -= self.data_total["Humidity"].min()
            self.data_total["Humidity"][:] /= self.data_total["Humidity"].max()
            # Wind Speed
            self.data_total["Wind Speed"][:] -= self.data_total["Wind Speed"].min()
            self.data_total["Wind Speed"][:] /= self.data_total["Wind Speed"].max()
            # general diffuse flows
            self.data_total["general diffuse flows"][:] -= self.data_total["general diffuse flows"].min()
            self.data_total["general diffuse flows"][:] /= self.data_total["general diffuse flows"].max()
            # diffuse flows
            self.data_total["diffuse flows"][:] -= self.data_total["diffuse flows"].min()
            self.data_total["diffuse flows"][:] /= self.data_total["diffuse flows"].max()
            # Zone 1 Power Consumption
            self.data_total["Zone 1 Power Consumption"] = self.data_total["Zone 1 Power Consumption"].fillna(0)
            self.data_total["Zone 1 Power Consumption"][:] -= self.data_total["Zone 1 Power Consumption"].min()
            self.data_total["Zone 1 Power Consumption"][:] /= self.data_total["Zone 1 Power Consumption"].max()
            # Zone 2 Power Consumption
            self.data_total["Zone 2  Power Consumption"] = self.data_total["Zone 2  Power Consumption"].fillna(0)
            self.data_total["Zone 2  Power Consumption"][:] -= self.data_total["Zone 2  Power Consumption"].min()
            self.data_total["Zone 2  Power Consumption"][:] /= self.data_total["Zone 2  Power Consumption"].max()
            # Zone 3 Power Consumption
            self.data_total["Zone 3  Power Consumption"] = self.data_total["Zone 3  Power Consumption"].fillna(0)
            self.data_total["Zone 3  Power Consumption"][:] -= self.data_total["Zone 3  Power Consumption"].min()
            self.data_total["Zone 3  Power Consumption"][:] /= self.data_total["Zone 3  Power Consumption"].max()

            # Create Three Datasets
            self.data_total_1 = self.data_total.drop(["Zone 2  Power Consumption", "Zone 3  Power Consumption"], axis=1)
            self.data_total_2 = self.data_total.drop(["Zone 1 Power Consumption", "Zone 3  Power Consumption"], axis=1)
            self.data_total_3 = self.data_total.drop(["Zone 1 Power Consumption", "Zone 2  Power Consumption"], axis=1)


        # for x in self.columns:
            #     print(x + "  -  ") #+ ": " + str(len(self.data_total[x].unique())))
            # print(len(self.data_total['CO2(tCO2)'].unique()))

        else:
            print("Already has Data")

    def get_xy_train(self):
        return
        # self.x_train = self.data_total.drop(["traffic_volume", "date_time"], axis=1)
        # self.data_total["traffic_volume"][:] -= self.data_total["traffic_volume"].min()
        # self.data_total["traffic_volume"][:] /= self.data_total["traffic_volume"].max()
        # self.y_train = self.data_total["traffic_volume"]

    def test(self):
        print(self.y_train)

    def clean_csv(self, file):
        if os.path.exists(file):
            temp = "tr -d '" + '"' + "' < " + file + " > clean_" + file
            os.system(temp)
            temp = "mv clean_" + file + " " + file
            os.system(temp)


tetuan = Tetuan()
tetuan.get_data()
print(tetuan.data_total.head())
tetuan.data_total_1.to_csv("tetuan_data_zone1.csv", index=False)
tetuan.clean_csv("tetuan_data_zone1.csv")
tetuan.data_total_2.to_csv("tetuan_data_zone2.csv", index=False)
tetuan.clean_csv("tetuan_data_zone2.csv")
tetuan.data_total_3.to_csv("tetuan_data_zone3.csv", index=False)
tetuan.clean_csv("tetuan_data_zone3.csv")

os.system("head tetuan_data_zone1.csv")
os.system("head tetuan_data_zone2.csv")
os.system("head tetuan_data_zone3.csv")

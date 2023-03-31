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

        self.x_train_zone1 = None
        self.y_train_zone1 = None
        self.x_test_zone1 = None
        self.y_test_zone1 = None

        self.x_train_zone2 = None
        self.y_train_zone2 = None
        self.x_test_zone2 = None
        self.y_test_zone2 = None

        self.x_train_zone3 = None
        self.y_train_zone3 = None
        self.x_test_zone3 = None
        self.y_test_zone3 = None

    def split_data(self, rand_state=7):
        self.x_train_zone1, self.x_test_zone1, self.y_train_zone1, self.y_test_zone1 = train_test_split(self.x_train_zone1, self.y_train_zone1, test_size=0.3, random_state=rand_state)
        self.x_train_zone2, self.x_test_zone2, self.y_train_zone2, self.y_test_zone2 = train_test_split(self.x_train_zone2, self.y_train_zone2, test_size=0.3, random_state=rand_state)
        self.x_train_zone3, self.x_test_zone3, self.y_train_zone3, self.y_test_zone3 = train_test_split(self.x_train_zone3, self.y_train_zone3, test_size=0.3, random_state=rand_state)

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
        self.x_train_zone1 = self.data_total_1.drop(["Zone 1 Power Consumption"], axis=1)
        self.y_train_zone1 = self.data_total_1["Zone 1 Power Consumption"]

        self.x_train_zone2 = self.data_total_2.drop(["Zone 2  Power Consumption"], axis=1)
        self.y_train_zone2 = self.data_total_2["Zone 2  Power Consumption"]

        self.x_train_zone3 = self.data_total_3.drop(["Zone 3  Power Consumption"], axis=1)
        self.y_train_zone3 = self.data_total_3["Zone 3  Power Consumption"]

    def test(self):
        print(self.y_train_zone1)

    def clean_csv(self, file):
        if os.path.exists(file):
            temp = "tr -d '" + '"[] ' + "' < " + file + " > clean_" + file
            os.system(temp)
            temp = "mv clean_" + file + " " + file
            os.system(temp)


tetuan = Tetuan()
tetuan.get_data()
tetuan.get_xy_train()
tetuan.split_data()

path_str1 = "tetuan_data_zone1_"

tetuan.x_train_zone1.to_csv(path_str1 + "x_train.csv", index=False, header=False)
tetuan.clean_csv(path_str1 + "x_train.csv")

tetuan.x_test_zone1.to_csv(path_str1 + "x_test.csv", index=False, header=False)
tetuan.clean_csv(path_str1 + "x_test.csv")

tetuan.y_train_zone1.to_csv(path_str1 + "y_train.csv", index=False, header=False)
tetuan.clean_csv(path_str1 + "y_train.csv")

tetuan.y_test_zone1.to_csv(path_str1 + "y_test.csv", index=False, header=False)
tetuan.clean_csv(path_str1 + "y_test.csv")

path_str1 = "tetuan_data_zone2_"

tetuan.x_train_zone2.to_csv(path_str1 + "x_train.csv", index=False, header=False)
tetuan.clean_csv(path_str1 + "x_train.csv")

tetuan.x_test_zone2.to_csv(path_str1 + "x_test.csv", index=False, header=False)
tetuan.clean_csv(path_str1 + "x_test.csv")

tetuan.y_train_zone2.to_csv(path_str1 + "y_train.csv", index=False, header=False)
tetuan.clean_csv(path_str1 + "y_train.csv")

tetuan.y_test_zone2.to_csv(path_str1 + "y_test.csv", index=False, header=False)
tetuan.clean_csv(path_str1 + "y_test.csv")

path_str1 = "tetuan_data_zone3_"

tetuan.x_train_zone3.to_csv(path_str1 + "x_train.csv", index=False, header=False)
tetuan.clean_csv(path_str1 + "x_train.csv")

tetuan.x_test_zone3.to_csv(path_str1 + "x_test.csv", index=False, header=False)
tetuan.clean_csv(path_str1 + "x_test.csv")

tetuan.y_train_zone3.to_csv(path_str1 + "y_train.csv", index=False, header=False)
tetuan.clean_csv(path_str1 + "y_train.csv")

tetuan.y_test_zone3.to_csv(path_str1 + "y_test.csv", index=False, header=False)
tetuan.clean_csv(path_str1 + "y_test.csv")


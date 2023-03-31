import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from sklearn.model_selection import train_test_split


class Metro:

    def __init__(self):
        self.path = "/Users/hacker/PycharmProjects/UROP/Metro_Interstate_Traffic_Volume.csv"
        self.columns = ['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main',
                        'weather_description', 'date_time', 'traffic_volume']
        self.data_total = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def split_data(self, rand_state=7):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train,
                                                                                test_size=0.3, random_state=rand_state)

    def get_data(self):
        if self.data_total is None:
            self.data_total = pd.read_csv(self.path)
            self.data_total = self.data_total.drop(["date_time"], axis=1)


            # FIXME Change this to already have everything Normalized
            """
            holiday: 0 / 1
            temp:  Normalize 0.0 - 1.0
            rain_1h: Normalize 0.0 - 1.0
            snow_1h:  Normalize 0.0 - 1.0
            clouds_all: Normalize 0.0 - 1.0
            weather_main: Categorical - One Hot
            weather_description: Categorical - One Hot
            date_time: Delete
            traffic_volume: Predictor Value4
            """

            # temp_min = self.data_total["temp"].min()

            self.data_total["holiday"][self.data_total["holiday"] == "None"] = 0.0
            self.data_total["holiday"][self.data_total["holiday"] != 0.0] = 1.0

            self.data_total["temp"][:] -= self.data_total["temp"].min()
            self.data_total["temp"][:] /= self.data_total["temp"].max()
            self.data_total["rain_1h"][:] -= self.data_total["rain_1h"].min()
            self.data_total["rain_1h"][:] /= self.data_total["rain_1h"].max()
            self.data_total["snow_1h"][:] -= self.data_total["snow_1h"].min()
            self.data_total["snow_1h"][:] /= self.data_total["snow_1h"].max()
            self.data_total["clouds_all"][:] -= self.data_total["clouds_all"].min()
            self.data_total["clouds_all"][:] /= self.data_total["clouds_all"].max()
            self.data_total["traffic_volume"][:] -= self.data_total["traffic_volume"].min()
            self.data_total["traffic_volume"][:] /= self.data_total["traffic_volume"].max()

            self.data_total["weather_main"] = self.data_total["weather_main"].str.upper()

            weather_main_temp = len(self.data_total["weather_main"].unique())
            weather_main_uniq = self.data_total["weather_main"].unique()
            print(weather_main_uniq)

            for x in range(weather_main_temp):
                zeros_temp = np.zeros(weather_main_temp)
                zeros_temp[x] = 1.0
                for y in range(len(self.data_total["weather_main"])):
                    if self.data_total["weather_main"][y] == weather_main_uniq[x]:
                        self.data_total["weather_main"][y] = pd.Series(zeros_temp).tolist()

            self.data_total["weather_main"] = list(self.data_total["weather_main"])

            self.data_total["weather_description"] = self.data_total["weather_description"].str.upper()

            weather_desc_temp = len(self.data_total["weather_description"].unique())
            weather_desc_uniq = self.data_total["weather_description"].unique()
            print(weather_desc_uniq)

            for x in range(weather_desc_temp):
                zeros_temp = np.zeros(weather_desc_temp)
                zeros_temp[x] = 1.0
                for y in range(len(self.data_total["weather_description"])):
                    if self.data_total["weather_description"][y] == weather_desc_uniq[x]:
                        self.data_total["weather_description"][y] = pd.Series(zeros_temp).tolist()

            self.data_total["weather_description"] = list(self.data_total["weather_description"])

            # for x in range(len(self.data_total)):
            #     if self.data_total[].replace(to_replace=) == "None":
            #         self.data_total.loc[:, ("holiday", x)] = 0.0
            #     else:
            #         self.data_total.loc[:, ("holiday", x)] = 1.0

                # self.data_total["temp"][x]. =
                # print(self.data_total["temp"].head(30))


        else:
            print("Already has Data")

    def get_xy_train(self):
        self.x_train = self.data_total.drop(["traffic_volume"], axis=1)
        self.y_train = self.data_total["traffic_volume"]


    def test(self):
        print(self.y_train)

    def clean_csv(self, file):
        if os.path.exists(file):
            temp = "tr -d '" + '"[] ' + "' < " + file + " > clean_" + file
            os.system(temp)
            temp = "mv clean_" + file + " " + file
            os.system(temp)

# data = pd.read_csv("./Datasets_old/Metro_Interstate_Traffic_Volume.csv")
#
# # print(data.head(5))
# #
# for x in ['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main',
#           'weather_description', 'date_time', 'traffic_volume']:
#     print(x + ": " + str(len(data[x].unique())))
# print(x + ": ")
# print(data["rain_1h"].mean())

# encodings = prep.OneHotEncoder(categories='weather_main')


metro = Metro()
metro.get_data()
metro.get_xy_train()
metro.split_data()

path_str = "metro_data_"

metro.x_train.to_csv(path_str + "x_train.csv", index=False, header=False)
metro.clean_csv(path_str + "x_train.csv")

metro.x_test.to_csv(path_str + "x_test.csv", index=False, header=False)
metro.clean_csv(path_str + "x_test.csv")

metro.y_train.to_csv(path_str + "y_train.csv", index=False, header=False)
metro.clean_csv(path_str + "y_train.csv")

metro.y_test.to_csv(path_str + "y_test.csv", index=False, header=False)
metro.clean_csv(path_str + "y_test.csv")



# metro.test()
# print(metro.data_total["holiday"][:10])
# print(metro.data_total["temp"][:10])
# os.system("head metro_data.csv")

"""
holiday: 0 / 1
temp:  Normalize 0.0 - 1.0
rain_1h: Normalize 0.0 - 1.0
snow_1h:  Normalize 0.0 - 1.0
clouds_all: Normalize 0.0 - 1.0
weather_main: Categorical - One Hot
weather_description: Categorical - One Hot
date_time: Delete
traffic_volume: Predictor Value
"""

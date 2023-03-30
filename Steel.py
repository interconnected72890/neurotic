import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from sklearn.model_selection import train_test_split


class Steel:

    def __init__(self):
        self.path = "/Users/hacker/PycharmProjects/UROP/Steel_industry_data.csv"
        self.columns = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
                        'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)',
                        'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM',
                        'WeekStatus', 'Day_of_week', 'Load_Type']
        self.data_total = None
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
            self.data_total = self.data_total.drop(['date'], axis=1)

            """
            ****
        XXX Usage_kWh - Normalize PREDICTER VALUE
            ****
            
        XXX Lagging_Current_Reactive.Power_kVarh - Normalize
        XXX Leading_Current_Reactive_Power_kVarh - Normalize
        XXX CO2(tCO2) - Normalize
        XXX Lagging_Current_Power_Factor - Normalize
        XXX Leading_Current_Power_Factor - Normalize
        XXX NSM - Normalize
        XXX WeekStatus - Binary 0/1 
            Day_of_week - One Hot Encoding
            Load_Type -  One Hot Encoding
            
            """
            # Usage
            self.data_total["Usage_kWh"][:] -= self.data_total["Usage_kWh"].min()
            self.data_total["Usage_kWh"][:] /= self.data_total["Usage_kWh"].max()
            # Lagging_Current_Reactive
            self.data_total["Lagging_Current_Reactive.Power_kVarh"][:] -= self.data_total["Lagging_Current_Reactive.Power_kVarh"].min()
            self.data_total["Lagging_Current_Reactive.Power_kVarh"][:] /= self.data_total["Lagging_Current_Reactive.Power_kVarh"].max()
            # Leading_Current_Reactive_Power_kVarh
            self.data_total["Leading_Current_Reactive_Power_kVarh"][:] -= self.data_total["Leading_Current_Reactive_Power_kVarh"].min()
            self.data_total["Leading_Current_Reactive_Power_kVarh"][:] /= self.data_total["Leading_Current_Reactive_Power_kVarh"].max()
            # CO2(tCO2)
            self.data_total["CO2(tCO2)"][:] -= self.data_total["CO2(tCO2)"].min()
            self.data_total["CO2(tCO2)"][:] /= self.data_total["CO2(tCO2)"].max()
            # Lagging_Current_Power_Factor
            self.data_total["Lagging_Current_Power_Factor"][:] -= self.data_total["Lagging_Current_Power_Factor"].min()
            self.data_total["Lagging_Current_Power_Factor"][:] /= self.data_total["Lagging_Current_Power_Factor"].max()
            # Leading_Current_Power_Factor
            self.data_total["Leading_Current_Power_Factor"][:] -= self.data_total["Leading_Current_Power_Factor"].min()
            self.data_total["Leading_Current_Power_Factor"][:] /= self.data_total["Leading_Current_Power_Factor"].max()
            # NSM
            self.data_total["NSM"][:] -= self.data_total["NSM"].min()
            self.data_total["NSM"][:] /= self.data_total["NSM"].max()


            # WeekStatus
            self.data_total["WeekStatus"][self.data_total["WeekStatus"] == "Weekday"] = 0.0
            self.data_total["WeekStatus"][self.data_total["WeekStatus"] != 0.0] = 1.0


            self.data_total["Day_of_week"] = self.data_total["Day_of_week"].str.upper()


            day_of_week_temp = len(self.data_total["Day_of_week"].unique())
            day_of_week_uniq = self.data_total["Day_of_week"].unique()
            print(day_of_week_uniq)

            for x in range(day_of_week_temp):
                zeros_temp = np.zeros(day_of_week_temp)
                zeros_temp[x] = 1.0
                for y in range(len(self.data_total["Day_of_week"])):
                    if self.data_total["Day_of_week"][y] == day_of_week_uniq[x]:
                        self.data_total["Day_of_week"][y] = pd.Series(zeros_temp).tolist()


            self.data_total["Load_Type"] = self.data_total["Load_Type"].str.upper()

            self.data_total["Load_Type"] = list(self.data_total["Load_Type"])

            load_type_temp = len(self.data_total["Load_Type"].unique())
            load_type_uniq = self.data_total["Load_Type"].unique()
            print(load_type_uniq)

            for x in range(load_type_temp):
                zeros_temp = np.zeros(load_type_temp)
                zeros_temp[x] = 1.0
                for y in range(len(self.data_total["Load_Type"])):
                    if self.data_total["Load_Type"][y] == load_type_uniq[x]:
                        self.data_total["Load_Type"][y] = pd.Series(zeros_temp).tolist()

            self.data_total["Load_Type"] = list(self.data_total["Load_Type"])

            # for x in self.columns:
            #     print(x + ": " + str(len(self.data_total[x].unique())))
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


steel = Steel()
steel.get_data()
print(steel.data_total.head())
# metro.get_xy_train()
steel.data_total.to_csv("steel_data.csv", index=False)
steel.clean_csv("steel_data.csv")
# metro.test()
# print(metro.data_total["holiday"][:10])
# print(metro.data_total["temp"][:10])
os.system("head steel_data.csv")

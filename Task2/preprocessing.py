import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class PreProcessing:
    def __init__(self):
        self.df = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None

    def read_data(self, filename, features, classes):
        self.df = pd.read_csv(filename)
        # self.x = pd.read_csv(filename)
        self.x = self.df.loc[self.df[self.df.columns[-1]].isin(classes)]
        self.y = self.df.iloc[:, -1]
        self.y = pd.get_dummies(self.y, columns=['Class'], dtype=int)
        self.x = pd.DataFrame(self.x[features])

    def split_data(self, split_rate):
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, test_size=split_rate / 100, stratify=self.y, random_state=42)

        # Convert the arrays back to DataFrames
        self.x_train = pd.DataFrame(self.x_train)
        self.x_test = pd.DataFrame(self.x_test)
        self.y_train = pd.DataFrame(self.y_train)
        self.y_test = pd.DataFrame(self.y_test)

    def null_handel(self):
        if 'MinorAxisLength' in self.x_train.columns:
            self.x_train.fillna(np.mean(self.x_train['MinorAxisLength']), inplace=True)
            self.x_train = pd.DataFrame(self.x_train)


    def normalize_train_data(self):
        self.scaler = MinMaxScaler()
        self.x_train = self.scaler.fit_transform(self.x_train.iloc[:, 0:5])
        self.x_train = pd.DataFrame(self.x_train)

    def normalize_test_data(self):
        self.x_test = self.scaler.transform(self.x_test.iloc[:, 0:5])
        self.x_test = pd.DataFrame(self.x_test)


    def handel_outlier_with_column(self, column):
        col = self.df[column]
        upper_limit = col.mean() + 3 * col.std()
        lowe_limit = col.mean() - 3 * col.std()
        self.df = self.df.loc[(col < upper_limit) & (col > lowe_limit)]

    def handel_all_outliers(self):
        # handel outlier in roundnes column
        self.handel_outlier_with_column('roundnes')

        # handel outlier in MinorAxisLength column
        self.handel_outlier_with_column('MinorAxisLength')

        print("data:", len(self.df))
        print("********************************************************")

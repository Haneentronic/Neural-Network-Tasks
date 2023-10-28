import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class PreProcessing:
    def __init__(self):
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None

    def read_data(self, filename, features, classes):
        self.x = pd.read_csv(filename)
        self.x = self.x.loc[self.x[self.x.columns[-1]].isin(classes)]
        self.y = self.x.iloc[:, -1:]
        self.x = self.x[features]

    def split_data(self, split_rate):
        self.x_train, self.x_test, self.y_train, self.y_test =\
            train_test_split(self.x, self.y, test_size=split_rate/100, stratify=self.y.values)

    def normalize_train_data(self):
        self.scaler = MinMaxScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)

    def normalize_test_data(self):
        self.x_test = self.scaler.transform(self.x_test)

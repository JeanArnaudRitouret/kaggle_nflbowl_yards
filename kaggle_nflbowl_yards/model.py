import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import *
import tensorflow

class Model():
    
    def __init__(self):
        pass

    def load_data(self, DIR_PATH = os.getcwd()+'/kaggle_nflbowl_yards/data/'):
        self.train = pd.read_csv(DIR_PATH + 'train_processed.csv')
        self.y = pd.read_csv(DIR_PATH + 'target_yards.csv')
        pass

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.train, self.y, 
                                                                                test_size=0.2, 
                                                                                random_state=2)
        pass
    
    def construct_compile(self):
        self.model = Sequential()
        self.model.add(layers.Dense(30, input_dim=92, activation='relu')) 
        self.model.add(layers.Dense(20, activation='tanh'))
        self.model.add(layers.Dense(10, activation='relu')) 
        self.model.add(layers.Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam',metrics='accuracy')
        return self.model.summary()

    def fit_evaluate(self):
        self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=10)
        self.evaluation = self.model.evaluate(self.X_test, self.y_test)
        return self.evaluation

if __name__=='__main__':
    model = Model()
    model.load_data()
    model.train_test_split()
    
    summary = model.construct_compile()
    print(summary)
    
    evaluation = model.fit_evaluate()
    print(evaluation)

import os
import pandas as pd
import numpy as np
from datetime import datetime
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import *
import tensorflow
import joblib
from google.cloud import storage


BUCKET_NAME = 'kaggle-nfl-bowl-yards'
BUCKET_TRAIN_DATA_PATH = 'data/train_processed.csv'
BUCKET_TARGET_DATA_PATH = 'data/train_processed.csv'
STORAGE_LOCATION = 'models/model_softmax.joblib'

class Model():
    
    def __init__(self):
        pass

    def load_data(self):
        #DIR_PATH = os.getcwd()+'/kaggle_nflbowl_yards/data/'
        self.train = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
        self.y = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TARGET_DATA_PATH}")
        pass

    def scale_data(self):
        scaler = MinMaxScaler()
        self.train = scaler.fit_transform(self.train)
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
        self.model.add(layers.Dense(199, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics='accuracy')
        return self.model.summary()

    def train_model(self):
        print(self.X_train)
        print(self.y_train)
        self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=10)
        self.evaluation = self.model.evaluate(self.X_test, self.y_test)
        return self.evaluation

    def upload_model_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model_softmax.joblib')

    def save_model(self):
        joblib.dump(self.model, 'model_softmax.joblib')
        self.upload_model_to_gcp()

if __name__=='__main__':
    model = Model()
    print('instantiate model')
    model.load_data()
    print('load data')
    model.scale_data()
    print('scale data')
    model.train_test_split()
    print('train test split')
    summary = model.construct_compile()
    print('construct compile')
    print(summary)
    
    evaluation = model.train_model()
    print(evaluation)

    model.save_model()

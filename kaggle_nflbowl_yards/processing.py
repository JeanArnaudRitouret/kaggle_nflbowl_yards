Class Processor():

    def __init__():
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

    def load_data(DATA_PATH = '../raw_data'):
        self.train = pd.read_csv(f'{DATA_PATH}/train.csv')
        self.train_cat = self.train.select_dtypes(include=['object'])
    
    # transforms serie into a binary type if team plays at home or away
    def proc_team(x):
        return 1 if x == 'home' else 0
    
    # transforms clock timevalues into minutes
    def proc_gameclock(x):
        min,sec,msec = x.split(':')
        return int(min) + int(sec)/60 + int(msec)/3600

    # creates a binary type column if possession happens in the field position of the team
    def possession_in_fieldPosition(df):
        df['PossessionInFieldPosition'] = df.FieldPosition == df.PossessionTeam
        df['PossessionInFieldPosition'] = df['PossessionInFieldPosition'].apply(lambda x : 1 if x else 0)
        return df.drop(columns = ['FieldPosition','PossessionTeam'])
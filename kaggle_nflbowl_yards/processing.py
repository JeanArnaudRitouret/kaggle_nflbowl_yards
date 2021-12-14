import os
import pandas as pd
import numpy as np
from datetime import datetime
import math
from sklearn.feature_extraction import DictVectorizer

class Processor:
    
    def __init__(self):
        self.dict_wind_direction = {
                                        'east': 'e', 
                                        'north': 'n', 
                                        'northwest': 'nw',
                                        'southwest': 'sw', 
                                        'northeast': 'ne', 
                                        'south': 's', 
                                        'west-southwest': 'wsw', 
                                        'south southeast': 'sse', 
                                        'west': 'w', 
                                        'northeast': 'ne', 
                                        'w-nw': 'wnw', 
                                        'south southwest': 'ssw', 
                                        'southeast': 'se',
                                        'west northwest': 'wnw',
                                        'east north east': 'ene', 
                                        'east southeast': 'ese',
                                        'north east': 'ne', 
                                        'north/northwest': 'nnw',
                                        'n-ne': 'nne', 
                                        'w-sw': 'wsw', 
                                        's-sw': 'ssw', 
                                        'south west': 'sw', 
                                        'south, southeast': 'sse', 
                                        'southerly': 's'
        }

    def load_data(self, DATA_PATH = os.getcwd()+'/raw_data/train.csv'):
        data = pd.read_csv(DATA_PATH)
        self.y = self.create_multiclass_target(data)
        self.train = data.drop(columns='Yards')
        return self.train
    
    def create_multiclass_target(self, data):
        a = np.zeros((data.shape[0],199))
        for i in range(a.shape[0]):
            for j in range(data.Yards[i]+99, 199):
                a[i,j] = 1
        return pd.DataFrame(a)

    # transforms serie into a binary type if team plays at home or away
    def proc_team(self, x):
        return 1 if x == 'home' else 0
    
    # transforms clock timevalues into minutes
    def proc_gameclock(self, x):
        min,sec,msec = x.split(':')
        return int(min) + int(sec)/60 + int(msec)/3600

    # creates a binary type column if possession team has the ball in its field position
    # FIeldPosition and PossessionTEam columns are dropped
    def possession_in_fieldPosition(self, df):
        df['PossessionInFieldPosition'] = df.FieldPosition == df.PossessionTeam
        df['PossessionInFieldPosition'] = df['PossessionInFieldPosition'].apply(lambda x : 1 if x else 0)
        return df.drop(columns = ['FieldPosition','PossessionTeam'])

    # One Hot Encoding of the offense formation column
    def oneHotEncoding_offense_formation(self, df):
        offense_formation_dummies = pd.get_dummies(df.OffenseFormation, columns=df.OffenseFormation.unique())
        return pd.concat([df.drop('OffenseFormation', axis=1),offense_formation_dummies], axis=1)
    
    # One Hot Encoding of the positions
    def oneHotEncoding_position(self, df):
        position_dummies = pd.get_dummies(df.Position, columns=df.Position.unique())
        return pd.concat([df.drop('Position', axis=1),position_dummies], axis=1)

    #creates a binary column if the play direction is left or right
    def proc_play_direction(self, x):
        return 1 if x == 'left' else 0
    
    # creates time delta series for handoff and players' age
    def proc_time_handoff_snap_and_player_age(self, df):
        seconds_in_year = 3600*24*365.25
        df['TimeDeltaHandoff'] = (df.TimeHandoff.apply(lambda x : datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%fZ'))
                            - df.TimeSnap.apply(lambda x : datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%fZ')))
        df['TimeDeltaHandoff'] = df['TimeDeltaHandoff'].apply(lambda x : x.total_seconds()/seconds_in_year)
        df['PlayerAge'] = (df.TimeHandoff.apply(lambda x : datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%fZ'))
                            - df.PlayerBirthDate.apply(lambda x : datetime.strptime(x,'%m/%d/%Y')))
        df['PlayerAge'] = df['PlayerAge'].apply(lambda x : x.total_seconds()/seconds_in_year)
        return df.drop(columns=['TimeHandoff','TimeSnap','PlayerBirthDate'])

    # format players' height into meters
    def proc_player_height(self, x):
        return float(f"{x.split('-')[0]}.{x.split('-')[1]}") * 30.48

    # standardize stadium types
    def convert_stadium_type_to_dict(self, text):
        stadium_type_dict = {}
        if str(text)=='nan':
            return stadium_type_dict
        if 'outdoor' in text.lower() or 'open' in text.lower() or 'heinz' in text.lower() \
        or 'ourdoor' in text.lower() or 'outdor' in text.lower():
            stadium_type_dict['outdoor'] = 1
        if ('indoor' in text.lower() and 'open' not in text.lower()) or 'closed' in text.lower():
            stadium_type_dict['indoor'] = 1
        if 'retr' in text.lower():
            stadium_type_dict['retractable'] = 1
        return stadium_type_dict
    
    # One hot encoding of standardized stadium types
    def oneHotEncoding_stadium_type(self, df):
        bow_stadium_type = df.StadiumType.apply(lambda x : self.convert_stadium_type_to_dict(x))
        vect = DictVectorizer(sparse=False)
        vectors_stadium_types = vect.fit_transform(bow_stadium_type)
        stadium_type_dummies = pd.DataFrame(vectors_stadium_types, columns=vect.get_feature_names())
        return pd.concat([df.drop('StadiumType', axis=1), stadium_type_dummies], axis=1)

    # standardize turfs type into Natural or Arftifical
    def convert_turf(self, x):
        return 'Natural' if x.lower() in ['grass','natural grass','natural','naturall grass'] else 'Artificial'

    # creates the serie with binary values if turf is natural or not
    def process_turf(self, df):
        df['IsTurfNatural'] = df.Turf.apply(lambda x : 1 if self.convert_turf(x)=='Natural' else 0)
        return df.drop('Turf', axis=1)

    # standardize game weather types
    def convert_game_weather_to_dict(self, text):
        game_weather_dict = {}
        if str(text)=='nan':
            return game_weather_dict
        if 'clear' in text.lower(): 
            game_weather_dict['clear'] = 1
        if 'warm' in text.lower():
            game_weather_dict['warm'] = 1
        if 'sun' in text.lower():
            game_weather_dict['sunny'] = 1
        if 'cloud' in text.lower() or 'coudy' in text.lower() or 'clouidy' in text.lower() or 'overcast' in text.lower():
            game_weather_dict['cloud'] = 1
        if 'indoor' in text.lower():
            game_weather_dict['indoor'] = 1
        if 'rain' in text.lower():
            game_weather_dict['rain'] = 1
        if 'shower' in text.lower():
            game_weather_dict['shower'] = 1
        if 'snow' in text.lower():
            game_weather_dict['snow'] = 1
        if 'cold' in text.lower():
            game_weather_dict['cold'] = 1
        if 'cool' in text.lower():
            game_weather_dict['cool'] = 1
        return game_weather_dict
    
    # One Hot Encoding of game weather types
    def oneHotEncoding_game_weather(self, df):
        bow_game_weather = df.GameWeather.apply(lambda x : self.convert_game_weather_to_dict(x))
        vect = DictVectorizer(sparse=False)
        vectors_game_weather = vect.fit_transform(bow_game_weather)
        game_weather_dummies = pd.DataFrame(vectors_game_weather, columns=vect.get_feature_names())
        return pd.concat([df.drop('GameWeather', axis=1), game_weather_dummies], axis=1)

    # transform wind speed into numerical values and use average if many values are given
    def process_wind_speed(self, x):
        digits = [int(i) for i in str(x).lower().replace('mph','').replace('.0','').replace('-',' ').split(' ') if i.isnumeric()]
        return sum(digits)/len(digits) if len(digits)>0 else 0
    
    # formating of the wind direction types
    def process_wind_direction(self, x):
        if isinstance(x,float) or x.isnumeric() or x.lower() in ['calm']:
            return ''
        return self.dict_wind_direction.get(x.lower().replace('from ',''), x.lower().replace('from ',''))
    
    #One Hot Encoding of the wind direction types
    def oneHotEncoding_wind_direction(self, df):
        wind_direction_dummies = pd.get_dummies(df.WindDirection.apply(lambda x : self.process_wind_direction(x)), columns=df.WindDirection.unique())
        return pd.concat([df.drop('WindDirection', axis=1), wind_direction_dummies], axis=1)
    
    def drop_categorical_features(self, df):
        return df.drop(columns=['DisplayName','OffensePersonnel','DefensePersonnel', 'PlayerCollegeName','HomeTeamAbbr','VisitorTeamAbbr','Stadium','Location'], axis=1)

    # creates binary values serie if row is rusher player or not
    def process_is_rusher(self, df):
        df['IsRusher'] = df.NflId == df.NflIdRusher
        df['IsRusher'] = df['IsRusher'].apply(lambda x : 1 if x else 0)
        return df.drop(columns=['NflId','NflIdRusher'])

    # replace null values with mean
    def proc_orientation(self, df):
        df.Orientation = df.Orientation.fillna(df.Orientation.mean())
        return df
    
    # replace null values with mean
    def proc_dir(self, df):
        df.Dir = df.Dir.fillna(df.Dir.mean())
        return df
    
    # replace null values with mean
    def proc_defenders_box(self, df):
        df.DefendersInTheBox = df.DefendersInTheBox.fillna(df.DefendersInTheBox.mean())
        return df
    
    # replace null values with mean
    def proc_temperature(self, df):
        df.Temperature = df.Temperature.fillna(df.Temperature.mean())
        return df
        
    # replace null values with mean
    def proc_humidity(self, df):
        df.Humidity = df.Humidity.fillna(df.Humidity.mean())
        return df
        
    def drop_numerical_features(self, df):
        return df.drop(columns=['GameId', 'PlayId', 'JerseyNumber','Season'], axis=1)
    
    def process_features(self, df_source):
        df = df_source.copy()
        df.Team = df.Team.apply(lambda x : self.proc_team(x))
        df.GameClock = df.GameClock.apply(lambda x : self.proc_gameclock(x))
        df = self.possession_in_fieldPosition(df)
        df = self.oneHotEncoding_offense_formation(df)
        df = self.oneHotEncoding_position(df)
        df.PlayDirection = df.PlayDirection.apply(lambda x : self.proc_play_direction(x))
        df = self.proc_time_handoff_snap_and_player_age(df)
        df.PlayerHeight = df.PlayerHeight.apply(lambda x : self.proc_player_height(x))
        df = self.oneHotEncoding_stadium_type(df)
        df = self.process_turf(df)
        df = self.oneHotEncoding_game_weather(df)
        df.WindSpeed = df.WindSpeed.apply(lambda x : self.process_wind_speed(x))
        df = self.oneHotEncoding_wind_direction(df)
        df = self.process_is_rusher(df)
        df = self.drop_categorical_features(df)
        df = self.proc_orientation(df)
        df = self.proc_dir(df)
        df = self.proc_defenders_box(df)
        df = self.proc_temperature(df)
        df = self.proc_humidity(df)
        df = self.drop_numerical_features(df)
        return df
    
    def save_data(self, df, DIR_PATH = os.getcwd()+'/kaggle_nflbowl_yards/data/'):
        df.to_csv(DIR_PATH + 'train_processed.csv')
        self.y.to_csv(DIR_PATH + 'target_processed.csv')

    
if __name__=='__main__':
    '''
    processor = Processor()
    train = processor.load_data()
    print(train)
    train_proc = processor.process_features(train)
    print(train_proc)
    processor.save_data(train_proc)
    '''

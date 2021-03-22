from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from itertools import chain, combinations
import os
import numpy as np

SECONDS_IN_HOUR = 3600
SECONDS_IN_MINUTE = 60

def getSheet(name):
    """Read a csv into a Pandas DataFrame"""
    
    csv = pd.read_csv(name)
    return pd.DataFrame(csv)

# locations returns, given the number of sheets, 
# the lower and upper bounds for each trial to 
# select from. 
def locations(name):
    csv = pd.read_csv(name)
    df = pd.DataFrame(csv)[['filename', 'start_stamp', 'start_frames_after', 'end_stamp', 'end_frames_after']]
    i = list(np.where(df['filename'] == 'Subject_02_P2_Zone12_T1')[0]) + list(np.where(df['filename'] == 'Subject_02_P2_Sit_T1')[0])
    df = df.drop(i).reset_index(drop=True)
    return df

# seconds_in_day returns, given an epoch,
# the number of seconds since the start
# of that day.
def seconds_in_day(epoch):
    d = datetime.utcfromtimestamp(epoch)
    return d.hour * SECONDS_IN_HOUR + d.minute * SECONDS_IN_MINUTE + d.second

def find_start_index(trial, start):
    stamps = trial.loc[:, 'Time_stp'].map(lambda x: seconds_in_day(x))
    start_index = (stamps >= start.loc['start_stamp']).idxmax()
    if stamps[0] <= start.loc['start_stamp']: # only add frames after if the lift started during the sequence
        start_index += start.loc['start_frames_after']
    end_index = (stamps >= start.loc['end_stamp']).idxmax()
    if stamps[0] <= start.loc['end_stamp']: # only add frames after if the lift started during the sequence
        end_index += start.loc['end_frames_after']
    
    return start_index, end_index

# process of converting the trials into vectors to train on
def preprocess(source, metadata, pad=True):
    sheets = []
    slices = locations(metadata)
    max_size = 0
    starts = []
    ends = []

    for i, trial in slices.iterrows():
        # extract columns
        df = getSheet(os.path.join(source, trial.loc['filename'] + '.csv'))
        start_index, end_index = find_start_index(df, trial)

        starts.append(start_index)
        ends.append(end_index)
        
        # remove timestamps from last column
        df = df.iloc[:, :-1]
      
        if df.shape[0] > max_size:
            max_size=df.shape[0]

        sheets.append(df)

    if pad:
        # pad data to length of longest trial
        for i in range(len(sheets)):
            df = sheets[i]
            diff = max_size - df.shape[0]
            if diff > 0:
                end_zeros = pd.DataFrame(np.zeros((diff, df.shape[1])))
                end_zeros.columns = df.columns

                sheets[i] = df.append(end_zeros)

    return (sheets, slices.loc[:, 'filename'], max_size, starts, ends)
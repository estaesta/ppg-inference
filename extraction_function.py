# import os
# import pickle
# import math
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# from scipy.interpolate import UnivariateSpline
from preprocessing_tool.feature_extraction import *

# constants
WINDOW_IN_SECONDS = 120  # 120 / 180 / 300  

# fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}
# polar verity sense
fs_dict = {'BVP': 44}


def preprocess_all(ppg_signal):
    """
    Preprocess the ppg signal and extract features

    Args:
    - ppg_signal: numpy array of ppg signal

    Returns:
    - processed_data: DataFrame of extracted features
    - window_len: number of windows
    """
        
    sec = 12
    N = fs_dict['BVP']*sec  # one block : 10 sec
    overlap = int(np.round(N * 0.02)) # overlapping length
    overlap = overlap if overlap%2 ==0 else overlap+1

    BP, FREQ, TIME, ENSEMBLE = False, False, False, False

    # feat_names = None

    # (band-pass filter), noise elimination, and ensemble
    # NOISE = ['bp_time_ens']
    n = 'bp_ens'
    if 'bp' in n.split('_'):
        BP = True
    if 'time' in n.split('_'):
        TIME = True
    if 'ens' in n.split('_'):
        ENSEMBLE = True

    # for patient in subject_ids:
    print(f'Processing data ...')
    processed_data, window_len, bp_bvp, hr = make_data(ppg_signal, BP, ENSEMBLE)
    print(processed_data)
    # normalization
    sc = joblib.load('./model/scaler_tri.pkl')
    # sc = joblib.load('./model/scaler_tri_1swin.pkl')
    processed_data = sc.transform(processed_data)
    # sc = StandardScaler()
    # processed_data = sc.fit_transform(processed_data)
    print(f'Processed data shape: {processed_data.shape}')
    # print(f'Processed data: {processed_data}')

    return processed_data, window_len, bp_bvp, hr


def make_data(ppg_signal, BP, ENSEMBLE):
    """
    Preprocess the ppg signal and extract features

    Args:
    - ppg_signal: numpy array of ppg signal
    - BP: boolean value to apply band-pass filter
    - ENSEMBLE: boolean value to apply ensemble

    Returns:
    - samples: pandas dataframe of extracted features
    - window_len: number of windows
    """
    # not doing noise elimination when inference
    
    # norm type
    norm_type = 'std'

    df = extract_ppg_data(ppg_signal, norm_type)
    df_BVP = df.BVP


    #여기서 signal preprocessing 
    bp_bvp = butter_bandpassfilter(df_BVP, 0.5, 10, fs_dict['BVP'], order=2) # 0.5, 5 -> 0.5,10
    
    if BP:
        df['BVP'] = bp_bvp
        
    # not used even in original code
    # if FREQ:
    #     signal_one_percent = int(len(df_BVP) * 0.01)
    #     print(signal_one_percent)
    #     cutoff = get_cutoff(df_BVP[:signal_one_percent], fs_dict['BVP'])
    #     freq_signal = compute_and_reconstruction_dft(df_BVP, fs_dict['BVP'], sec, overlap, cutoff)
    #     df['BVP'] = freq_signal

    # ignore
    # if TIME:
    #     #temp_ths = [1.1,2.2,2.0,1.9] 
    #     temp_ths = [1.0,2.0,1.8,1.5] 
    #     clean_df = pd.read_csv('clean_signal_by_rate.csv',index_col=0)
    #     cycle = 15
    #
    #     fwd = moving_average(bp_bvp, size=3)
    #     bwd = moving_average(bp_bvp[::-1], size=3)
    #     bp_bvp = np.mean(np.vstack((fwd,bwd[::-1])), axis=0)
    #     df['BVP'] = bp_bvp
    #     
    #     signal_01_percent = int(len(df_BVP) * 0.001)
    #     print(signal_01_percent, int(clean_df.loc[subject_id]['index']))
    #     clean_signal = df_BVP[int(clean_df.loc[subject_id]['index']):int(clean_df.loc[subject_id]['index'])+signal_01_percent]
    #     ths = statistic_threshold(clean_signal, fs_dict['BVP'], temp_ths)
    #     len_before, len_after, time_signal_index = eliminate_noise_in_time(df['BVP'].to_numpy(), fs_dict['BVP'], ths, cycle)
    # 
    #     df = df.iloc[time_signal_index,:]
    #     df = df.reset_index(drop=True)
    #     #plt.figure(figsize=(40,20))
    #     #plt.plot(df['BVP'][:2000], color = 'b', linewidth=2.5)
    
    fwd = moving_average(bp_bvp, size=3)
    bwd = moving_average(bp_bvp[::-1], size=3)
    bp_bvp = np.mean(np.vstack((fwd,bwd[::-1])), axis=0)
    df['BVP'] = bp_bvp
    samples, hr = get_samples(df, BP, ENSEMBLE)
    return samples, len(samples), bp_bvp, hr

def extract_ppg_data(ppg_signal, norm_type=None):
    """
    Convert ppg signal to pandas dataframe and normalize

    Args:
    - ppg_signal: numpy array of ppg signal
    - norm_type: normalization type (std, minmax)

    Returns:
    - df: pandas dataframe of ppg signal
    """
    # Dataframes for each sensor type
    df = pd.DataFrame(ppg_signal, columns=['BVP'])
    
    # Adding indices for combination due to differing sampling frequencies
    df.index = [(1 / fs_dict['BVP']) * i for i in range(len(df))]

    # Change indices to datetime
    df.index = pd.to_datetime(df.index, unit='s')

    df.reset_index(drop=True, inplace=True)
    
    if norm_type == 'std':
        # std norm
        df['BVP'] = (df['BVP'] - df['BVP'].mean()) / df['BVP'].std()
    elif norm_type == 'minmax':
        # minmax norm
        df = (df - df.min()) / (df.max() - df.min())

    # Groupby
    # df = df.dropna(axis=0) # nan인 행 제거
    
    return df

def get_samples(data, ma_usage, ensemble):
    """
    Windowing the data and extract features

    Args:
    - data: pandas dataframe of ppg signal
    - ma_usage: boolean value to apply moving average
    - ensemble: boolean value to apply ensemble

    Returns:
    - samples: pandas dataframe of extracted features
    """
    # global WINDOW_IN_SECONDS

    samples = []

    window_len = fs_dict['BVP'] * WINDOW_IN_SECONDS  # 64*60 , sliding window: 0.25 sec (60*0.25 = 15)   
    sliding_window_len = int(fs_dict['BVP'] * WINDOW_IN_SECONDS * 0.25)
    
    winNum = 0
    
    i = 0
    while sliding_window_len * i <= len(data) - window_len:
        
         # 한 윈도우에 해당하는 모든 윈도우 담기,
        w = data[sliding_window_len * i: (sliding_window_len * i) + window_len]  
        # Calculate stats for window
        wstats, hr = get_window_stats_27_features(ppg_seg=w['BVP'].tolist(), window_length = window_len, ensemble = ensemble, ma_usage=ma_usage, fs = fs_dict['BVP'])
        winNum += 1
        
        if wstats == []:
            i += 1
            continue;
        # Seperating sample and label
        x = pd.DataFrame(wstats, index = [i])
    
        samples.append(x)
        i += 1

    return pd.concat(samples), hr

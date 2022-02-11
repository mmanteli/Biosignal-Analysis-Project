from unicodedata import numeric
import pandas as pd
import glob
import os
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, lfilter
from sklearn import preprocessing as prep
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


eeg_frequencies = [0.5,4,8,13,35]
labels_dir = {1:"Awake", 2:"REM", 3:"N1", 4:"N2", 5:"N3", 6:"N4"}
PATH = os.path.join(os.getcwd(), 'data', 'train')
fs = 100


def read_data(PATH):
    raw_data = []

    for label in glob.glob(PATH+"/*"):
        for file in glob.glob((label+"/*.csv")):
            data = pd.read_csv(file)
            #data["label2"] = [label.split(os.sep)[-1]]*len(data)
            data_label = {value : key for (key, value) in labels_dir.items()}[label.split(os.sep)[-1]]
            data["label"] = [data_label]*len(data)
            raw_data.append(data)

    return raw_data

# filtteroi jokainen raw_data jonon jäsen



# etsi tarvittavat ominaisuudet ja tee uusi taulukko
# indeksi   feat1   feat2   ... featn   label


def filter_butterworth(raw_data, num_channels, fc, btype):
    df_filtered = pd.DataFrame()
    print("         ")
    print(raw_data['EEG Fpz-Cz'].to_numpy())
    print("         ")
    Wn = np.array(fc)/(fs/2.0)
    b, a = butter(2, Wn, btype =btype, analog=False)
    #df_filtered['EEG Fpz-Cz'] = filtfilt(b, a, raw_data['EEG Fpz-Cz'].to_numpy())
    #df_filtered['EEG Pz-Oz'] = filtfilt(b, a, raw_data['EEG Pz-Oz'].to_numpy())
    filtfilt(b, a, raw_data['EEG Fpz-Cz'].to_numpy())
    #df_filtered['label'] = raw_data['label']
    return df_filtered


def butter_bandpass_filter(lowcut, highcut, freq, order, data):
    df_filtered = pd.DataFrame()
    for label in ['EEG Fpz-Cz', 'EEG Pz-Oz']:
        x = data[label].to_numpy()
        sampling_rate = 0.5*freq
        b,a = butter(order, [lowcut/sampling_rate, highcut/sampling_rate], btype = 'band')
        y = filtfilt(b,a,x)
        df_filtered[label] = y
    df_filtered['label'] = data['label']
    return df_filtered


def feature_extraction(sig, hist_limit, hist_length):
    """
    extracts features from EEG signal array: [EEG1, EEG2, Label]
    returns a vector with means, stds, histograms of PSD and label

    hist_limit: highest frequency in histograms
    """

    
    feature_array = []

    #numeric_sig = sig.drop('label', axis = 1)
    numeric_sig = sig.drop('label', axis = 1)
    # signal mean and standard deviation
    feature_array.append(np.mean(numeric_sig).to_list())
    #feature_array = [item for sublist in feature_array for item in sublist]
    feature_array.append(np.std(numeric_sig).to_list())
    feature_array = [item for sublist in feature_array for item in sublist]
    
    # entropy or energy of the signal?
    # measures of activity?

    # interquartile range
    #a,b = np.percentile(sig, [75 ,25])
    #iqr = b-a
    #feature_array.append(iqr)

    # standardize for frequency extraction
    transformed_sig = prep.scale(numeric_sig)
    
    #print(type(transformed_sig))
    PSD_binned_sum = [] 

    # Power spectral density
    for column in range(len(transformed_sig[0])):
        eeg = transformed_sig[:,column]

        # Check this (windowsize) maybe larger?
        frequencies , PSD = signal.welch(eeg, fs=100)

        # different binning systems for different waves
        # differnent features of each bin, now only integral, ratios of delta / beta /alpha etc.
        # normalize wrt bin length
        freqs_arr = np.array(eeg_frequencies)
        
        for i in range(freqs_arr.shape[0]-1): 
            PSD_binned_sum.append (np.sum(PSD[np.where( (frequencies >= freqs_arr[i] ) & 
                                                    ( frequencies < freqs_arr[i+1] ) )]) )
        
    # append the results
    for bins in PSD_binned_sum:
      feature_array.append(bins)

    feature_array.append(sig['label'][0])
    return feature_array

def filter_data(raw_data):
    filtered_data = []
    for data in raw_data:
        f_data = butter_bandpass_filter(0.1, 35, 100, 2, data)
        filtered_data.append(f_data)
    return filtered_data


def extract_data(filtered_data):

    final_data = []

    for f_data in filtered_data:
        
        final_data.append(feature_extraction(f_data, hist_limit=40, hist_length=5))

    columns = ['mean_Cz', 'mean_Oz', 'std_Cz', 'std_Oz']
    columns += ['hist_Cz'+"_"+str(i) for i in ["delta", "theta", "alpha", "beta"]]
    columns += ['hist_Oz'+"_"+str(i) for i in  ["delta", "theta", "alpha", "beta"]]
    columns.append('label')

    preprocessed_data = pd.DataFrame(data= final_data, columns=columns)

    return preprocessed_data



raw_data = read_data(PATH)
print("DATA LUETTU")

filtered_data = filter_data(raw_data)
print("DATA FILTTERÖITY")
preprocessed_data = extract_data(filtered_data)
print("DATA PROSESSOITU")

print(preprocessed_data)




#______________________________________________________________________________________________________________________

# train ja val setit
X = preprocessed_data.drop('label',axis = 1)
Y = preprocessed_data['label']

X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2)#, stratify=preprocessed_data['label'].to_numpy())



knn = KNeighborsClassifier(n_neighbors=7)

# fit the train data
knn.fit(X_train, Y_train)

pred =knn.predict(X_val)

# Then, use confusion_matrix() with correct labels Y_test and the predictions
print(confusion_matrix(Y_val, pred))
print(classification_report(Y_val, pred))

print("Classification accuracy with knn.score(): ",knn.score(X_val, Y_val))


# graafinen esitys -koodi
    


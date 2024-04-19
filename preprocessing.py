import sys

sys.path.append("../src/")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from src.data import load_signals, load_annotations, annotation_to_30s_labels
from scipy.signal import butter, lfilter
from tsflex.processing import SeriesPipeline, SeriesProcessor
import antropy as ant
import scipy.stats as ss
from yasa import bandpower
import pickle
import scipy.stats as ss
from tsflex.features import (
    FeatureCollection,
    FuncWrapper,
    MultipleFeatureDescriptors,
    FuncWrapper,
)
from tsflex.features.integrations import tsfresh_settings_wrapper


def wrapped_higuchi_fd(x):
    x = np.array(x, dtype="float64")
    return ant.higuchi_fd(x)


bands = [
    (0.4, 1, "sdelta"),
    (1, 4, "fdelta"),
    (4, 8, "theta"),
    (8, 12, "alpha"),
    (12, 16, "sigma"),
    (16, 30, "beta"),
]
bandpowers_ouputs = [b[2] for b in bands] + ["TotalAbsPow"]


def wrapped_bandpowers(x, sf, bands):
    return bandpower(x, sf=sf, bands=bands).values[0][:-2]


def butter_bandpass_filter(sig, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, sig)
    return y


def extract_raw_windows(data_series: pd.Series, timestamps: pd.DatetimeIndex):
    """
    Extract 30-second windows from a pd.Series based on a list of start times (DatetimeIndex).

    Parameters:
    - data_series (pd.Series): Time series data indexed by timestamp from which to extract segments.
    - timestamps (pd.DatetimeIndex): DatetimeIndex containing the start times for each window.

    Returns:
    - pd.DataFrame: DataFrame containing all segments, each indexed by the original timestamp index,
                    with an additional 'Window' column indicating the window number.
    """
    # Duration of each window
    duration = pd.Timedelta(seconds=30)

    # Using list comprehension to create a DataFrame for each window
    segments = [data_series[start:start + duration] for start in timestamps]
    return segments

def extract_raw_data(self, data: pd.DataFrame, window_size: pd.Timedelta, stride: pd.Timedelta):
    """
    Extracts raw data segments based on the specified window size and stride.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the time series data.
    - window_size (pd.Timedelta): The size of the window for which to extract data.
    - stride (pd.Timedelta): The stride between windows.

    Returns:
    - List[pd.DataFrame]: A list of DataFrame segments containing the raw data for each window.
    """
    start_time = data.index[0]
    end_time = data.index[-1]
    current_time = start_time
    segments = []

    while current_time + window_size <= end_time:
        segment = data.loc[current_time:current_time + window_size]
        segments.append(segment)
        current_time += stride

    return segments


signals_types = [
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    # "EOG horizontal",  # All have the same sampling rate (100 Hz)
    # "EMG submental",  # Different sampling rate: 100 Hz for telemetry & 1 Hz for cassette
]

eeg_bandpass = SeriesProcessor(
    function=butter_bandpass_filter,
    series_names=["EEG Fpz-Cz", "EEG Pz-Oz",
                  # "EOG horizontal"
                  ],
    lowcut=0.4,
    highcut=30,
    fs=100,
)

process_pipe = SeriesPipeline(
    [
        eeg_bandpass,
    ]
)

tsfresh_settings = {
    "fft_aggregated": [
        {"aggtype": "centroid"},
        {"aggtype": "variance"},
        {"aggtype": "skew"},
        {"aggtype": "kurtosis"},
    ],
    "fourier_entropy": [
        {"bins": 2},
        {"bins": 3},
        {"bins": 5},
        {"bins": 10},
        {"bins": 30},
        {"bins": 60},
        {"bins": 100},
    ],
    "binned_entropy": [
        {"max_bins": 5},
        {"max_bins": 10},
        {"max_bins": 30},
        {"max_bins": 60},
    ],
}

time_funcs = [
                 np.std,
                 ss.iqr,
                 ss.skew,
                 ss.kurtosis,
                 ant.num_zerocross,
                 FuncWrapper(
                     ant.hjorth_params, output_names=["horth_mobility", "hjorth_complexity"]
                 ),
                 wrapped_higuchi_fd,
                 ant.petrosian_fd,
                 ant.perm_entropy,
             ] + tsfresh_settings_wrapper(tsfresh_settings)

freq_funcs = [
    FuncWrapper(wrapped_bandpowers, sf=100, bands=bands, output_names=bandpowers_ouputs)
]

time_feats = MultipleFeatureDescriptors(
    time_funcs,
    ["EEG Fpz-Cz", "EEG Pz-Oz",
     # "EOG horizontal",
     # "EMG submental"
     ],
    windows=["30s"],
    strides="30s",
)
freq_feats = MultipleFeatureDescriptors(
    freq_funcs,
    ["EEG Fpz-Cz", "EEG Pz-Oz",
     # "EOG horizontal"
     ],
    windows=["30s"],
    strides="30s",
)

feature_collection = FeatureCollection([time_feats, freq_feats])

data_folder = "./data/sleep-edf-database-expanded-1.0.0/"

dfs = []
sub_folder = "sleep-cassette"
sorted_files = sorted(os.listdir(data_folder + sub_folder))
psg_hypnogram_files = [(p, h) for p, h in zip(sorted_files[::2], sorted_files[1:][::2])]
df_files = pd.DataFrame(psg_hypnogram_files, columns=["psg_file", "label_file"])
df_files["subfolder"] = sub_folder

# Because for the SC study patients were monitored for 2 consecutive nights
df_files["patient_id"] = df_files.psg_file.apply(lambda f: f[:5])
patient_ids = [i for i in df_files["patient_id"]]
unique_ids = np.unique(patient_ids)

# split into train test 8:2 across subjects
# train_sbj, test_sbj = train_test_split(unique_ids, test_size=0.2, random_state=0)
#
# df_files_train = df_files[df_files.patient_id.isin(train_sbj)]
#
# df_files_test = df_files[df_files.patient_id.isin(test_sbj)]

for sub_folder, psg_file, hypnogram_file in tqdm(
        zip(df_files.subfolder, df_files.psg_file, df_files.label_file)
):
    file_folder = data_folder + sub_folder + "/"
    # Load the data, process the data and extract features

    data = load_signals(file_folder + psg_file, retrieve_signals=signals_types)

    annotations = load_annotations(file_folder + hypnogram_file, file_folder + psg_file)
    annotations = annotation_to_30s_labels(annotations)
    data_processed = process_pipe.process(data)

    df_feat = feature_collection.calculate(
        data_processed, return_df=True, window_idx="begin"
    ).astype("float32")

    eeg_signals = [d.name for d in data_processed if "EEG" in d.name]
    bands = ["alpha", "beta", "sdelta", "fdelta", "sigma", "theta"]
    for eeg_sig in eeg_signals:
        eeg_bands = [c for c in df_feat.columns if c.startswith(eeg_sig) and c.split("__")[1] in bands]
        windows = sorted(set(b.split("__")[-1] for b in eeg_bands))
        for window in windows:
            # Select the spectral powers
            delta = df_feat["__".join([eeg_sig, "sdelta", window])] + df_feat["__".join([eeg_sig, "fdelta", window])]
            fdelta_theta = df_feat["__".join([eeg_sig, "fdelta", window])] + df_feat[
                "__".join([eeg_sig, "theta", window])]
            alpha = df_feat["__".join([eeg_sig, "alpha", window])]
            beta = df_feat["__".join([eeg_sig, "beta", window])]
            theta = df_feat["__".join([eeg_sig, "theta", window])]
            sigma = df_feat["__".join([eeg_sig, "sigma", window])]
            # Calculate the ratios
            df_feat["__".join([eeg_sig, "fdelta+theta", window])] = fdelta_theta.astype("float32")
            df_feat["__".join([eeg_sig, "alpha/theta", window])] = (alpha / theta).astype("float32")
            df_feat["__".join([eeg_sig, "delta/beta", window])] = (delta / beta).astype("float32")
            df_feat["__".join([eeg_sig, "delta/sigma", window])] = (delta / sigma).astype("float32")
            df_feat["__".join([eeg_sig, "delta/theta", window])] = (delta / theta).astype("float32")



    # Add the labels (and reduce features to only data for which we have labels)

    df_feat = df_feat.merge(annotations, left_index=True, right_index=True)

    time_stamps = df_feat.index

    raw_windows = [extract_raw_windows(w, time_stamps) for w in data_processed]

    raw_data = []
    for ch in raw_windows:
        ch_data = []
        for w in ch:
            ch_data.append(w.values[:-1])
        raw_data.append(np.array(ch_data))
    raw_data = np.array(raw_data).swapaxes(0, 1)


    # Add the file name & folder
    df_feat["psg_file"] = psg_file
    df_feat["patient_id"] = psg_file[:5]

    patient_ids = df_feat["patient_id"].values
    patient_features = df_feat.values[:, :-3]
    labels = df_feat.values[:, -3]

    patient_id = np.unique(patient_ids)[0]
    filename = np.unique(df_feat["psg_file"].values)[0]

    # df_feat[data_processed[0].name + "_raw"] = raw_windows[0]
    # df_feat[data_processed[1].name + "_raw"] = raw_windows[1]
    # Collect the dataframes
    # df_feats += [df_feat]

    pickle.dump([raw_data, patient_features, labels, patient_ids], open('./features/' + filename[:8] + '.p', 'wb'))
    print(patient_id + '_' + filename + ' finished saving...')

# df_feats = pd.concat(df_feats)
# # df_feats.rename(columns={"description": "label"}, inplace=True)
# # df_feats.to_parquet("./features/sleep-edf__telemetry_features_ALL__30s.parquet")
#
print('finished saving')


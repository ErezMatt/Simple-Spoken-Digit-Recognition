import numpy as np
import librosa
import keras

def load_padded(file_path, duration, mode="constant"):
    signal, sr = librosa.load(file_path, duration=duration)

    num_expected_samples = int(sr * duration)

    if len(signal) < num_expected_samples:
        num_missing_samples = num_expected_samples - len(signal)
        signal = np.pad(signal, (0, num_missing_samples), mode=mode)
    return signal, sr

def exctract_melspectrogram(file_path, duration, n_fft=1024):
    signal, sr = load_padded(file_path, duration)
    ps = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft)
    return ps

def exctract_mfcc(file_path, duration, n_fft=1024, hop_length=512, n_mfcc=128):
    signal, sr = load_padded(file_path, duration)
    ps = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    return ps

def prepare_dataset(df, label_count, duration, feature_type='melspectrogram', n_fft=1024, hop_length=512, n_mfcc=128):
    """
    Create dataset of MFCCs or melspectrograms extracted from audio files

    :param df: pd.dataframe with audio file paths and labels
    :param label_count: Number of classes
    :param duration: sound duration in ms
    :param feature_type: string to specify data, can be melspectrogram or mfcc

    Parameter for librosa functions
    :param n_fft: length of FFT window
    :param hop_length: number of samples between frames
    :param n_mfcc: numbver of MFCCs
    :return: 2 np.arrays X that contains data and y with one-hot encoded labels
    """
    X = []
    y = []
    shapes = set()
    for row in df.itertuples():
        if feature_type=='melspectrogram':
            ps = exctract_melspectrogram(row.File_Path, duration=duration, n_fft=n_fft)
        elif feature_type=='mfcc':
            ps = exctract_mfcc(row.File_Path, duration=duration, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
        else:
            raise Exception(f'Unsupported feature_type {feature_type} should be: (melspectrogram, mfcc)')
        X.append(ps)
        y.append(keras.utils.to_categorical(row.Label, label_count))

        shapes.add(ps.shape)
    assert len(shapes) <= 1, 'Features size problem {shapes}'
    
    X = np.array(X)
    y = np.array(y)

    return X, y
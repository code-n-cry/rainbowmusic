import librosa
import pandas as pd
from numpy import mean, var


def featuring(sound, sr: int = 22050) -> pd.DataFrame:
    file, _ = librosa.effects.trim(sound)
    harmony, perceptr = librosa.effects.hpss(file)
    features_dict = {
            'chroma_stft_mean': mean(librosa.feature.chroma_stft(file)),
            'chroma_stft_var': var(librosa.feature.chroma_stft(file)),
            'rms_mean': mean(librosa.feature.rms(file)),
            'rms_var': var(librosa.feature.rms(file)),
            'spectral_centroid_mean': mean(librosa.feature.spectral_centroid(file)),
            'spectral_centroid_var': var(librosa.feature.spectral_centroid(file)),
            'spectral_bandwidth_var': var(librosa.feature.spectral_bandwidth(file)),
            'spectral_bandwidth_mean': mean(librosa.feature.spectral_bandwidth(file)),
            'rolloff_var': var(librosa.feature.spectral_rolloff(file)),
            'rolloff_mean': mean(librosa.feature.spectral_rolloff(file)),
            'zero_crossing_rate_mean': mean(librosa.feature.zero_crossing_rate(file)),
            'zero_crossing_rate_var': var(librosa.feature.zero_crossing_rate(file)),
            'harmony_mean': mean(harmony),
            'harmony_var': var(harmony),
            'perceptr_mean': mean(perceptr),
            'perceptr_var': var(perceptr),
            'tempo': librosa.beat.tempo(file, sr=sr)[0],
        }
    mfccs = librosa.feature.mfcc(file, n_mfcc=20)
    for i in range(len(mfccs)):
            features_dict[f'mfcc{i + 1}_mean'] = mean(mfccs[i])
            features_dict[f'mfcc{i + 1}_var'] = var(mfccs[i])
    return features_dict


def to_df(filename: str):
    df_dict = {}
    second_block = 3
    y, _ = librosa.load(filename)
    count_blocks = round(librosa.get_duration(y) / second_block)
    sr = 22050
    for i in range(count_blocks):
        sound, sr = librosa.load(
            path=filename,
            sr=sr,
            offset=(i * second_block),
            duration=second_block
        )
        features = featuring(sound, sr)
        for key in features:
            if key not in df_dict:
                df_dict[key] = []
            df_dict[key].append(features[key])
    return pd.DataFrame(df_dict)

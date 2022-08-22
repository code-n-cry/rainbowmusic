import librosa
import pandas as pd


def create_audio_features(sound, sr=22050):
    audio_file, _ = librosa.effects.trim(y=sound)
    audio_features = {}
    chromagram = librosa.feature.chroma_stft(y=audio_file, sr=sr)[0]
    audio_features['chroma_stft_mean'] = chromagram.mean()
    audio_features['chroma_stft_var'] = chromagram.var()
    rms = librosa.feature.rms(y=audio_file)[0]
    audio_features['rms_mean'] = rms.mean()
    audio_features['rms_var'] = rms.var()
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_file, sr=sr)[0]
    audio_features['spectral_centroid_mean'] = spectral_centroids.mean()
    audio_features['spectral_centroid_var'] = spectral_centroids.var()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_file, sr=sr)[0]
    audio_features['spectral_bandwidth_mean'] = spectral_bandwidth.mean()
    audio_features['spectral_bandwidth_var'] = spectral_bandwidth.var()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_file, sr=sr)[0]
    audio_features['rolloff_mean'] = spectral_rolloff.mean()
    audio_features['rolloff_var'] = spectral_rolloff.var()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_file)[0]
    audio_features['zero_crossing_rate_mean'] = zero_crossing_rate.mean()
    audio_features['zero_crossing_rate_var'] = zero_crossing_rate.var()
    harmony, perceptr = librosa.effects.hpss(y=audio_file)
    audio_features['harmony_mean'] = harmony.mean()
    audio_features['harmony_var'] = harmony.var()
    audio_features['perceptr_mean'] = perceptr.mean()
    audio_features['perceptr_var'] = perceptr.var()
    tempo, _ = librosa.beat.beat_track(y=audio_file, sr = sr)
    audio_features['tempo'] = tempo
    mfccs = librosa.feature.mfcc(y=audio_file, sr=sr)    
    for i in range(mfccs.shape[0]):
        audio_features[f'mfcc{i+1}_mean'] = mfccs[i].mean()
        audio_features[f'mfcc{i+1}_var'] = mfccs[i].var()
    return audio_features


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
        features = create_audio_features(sound, sr)
        for key in features:
            if key not in df_dict:
                df_dict[key] = []
            df_dict[key].append(features[key])
    return pd.DataFrame(df_dict)

import pandas as pd
import numpy as np
import tensorflow as tf
import typing
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import utils
from tensorflow.keras import backend
from tensorflow.keras import metrics
from sklearn import preprocessing
from sklearn import model_selection

SEED = 27
encoder = preprocessing.LabelEncoder()
tf.random.set_seed(SEED)
scaler = preprocessing.MinMaxScaler()
data = pd.read_csv('features_3_sec.csv', delimiter=',', encoding='utf-8')
cleaned_data = data.drop(['filename', 'length', 'label'], axis=1)
y = data['label']
x = cleaned_data
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,
                                              test_size=0.3, random_state=SEED)
x_validation, x_test, y_validation, y_test = model_selection.train_test_split(
    x_test, y_test, test_size=0.37, random_state=SEED)
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_validation = pd.DataFrame(scaler.transform(x_validation), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_train.columns)
y_train, y_validation, y_test = encoder.fit_transform(y_train), encoder.transform(y_validation), encoder.transform(y_test)
y_train, y_validation, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_validation, 10), utils.to_categorical(y_test, 10)
backend.clear_session()
model = models.Sequential([
    layers.InputLayer(input_shape=(x.shape[1])),
    layers.Dense(units=1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(units=512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(units=256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(units=128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=10, activation='softmax')
])


def train_model(model: models.Sequential, optimizer: str, epochs: int,
                metrics: list):
  model.compile(optimizer, loss='categorical_crossentropy', metrics=metrics)
  print(x_train.shape, y_train.shape)
  history = model.fit(
        x_train, y_train, 128, epochs,
        validation_data=(x_validation, y_validation), shuffle=True
  )
  model.save_weights('music.h5')
  return history


def plot_history(history):
    print("Max. Validation Accuracy", max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12, 6))
    plt.show()


plot_history(train_model(model, 'Adam', 150, ['accuracy']))
genres = ['Блюз', 'Классика', "Кантри", "Диско", "Хип-хоп", "Джаз", "Метал", "Поп", "Регги", "Рок"]
predict_data = pd.DataFrame({
    'Id': range(0, x_test.shape[0]),
    'Label': genres[np.argmax(model.predict(x_test), axis=1)]
})
print(predict_data)
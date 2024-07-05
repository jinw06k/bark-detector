import json
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

DATA_PATH = "data.json"

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data loaded")

    return  X, y


if __name__ == "__main__":

    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = keras.Sequential([

        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        keras.layers.Dense(256, activation='relu'),

        keras.layers.Dropout(0.3),

        keras.layers.Dense(256, activation='relu'),

        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation='relu'),

        keras.layers.Dense(1, activation='sigmoid')

    ])

    optimiser = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=30)

    score = model.evaluate(X_test, y_test)
    print(score)

    model.save("model.keras")

    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()

    # open('model.tflite', 'wb').write(tflite_model)

    # print("TensorFlow Lite model saved")

    
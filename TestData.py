import json
import numpy as np
import tensorflow as tf

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data loaded")

    return  X, y    


model = tf.keras.models.load_model('model.keras')
X_dog, y_dog = load_data("test.json")

i = 0
for x in X_dog:
    prediction = model.predict(x[np.newaxis, ...])

    print("Prediction: ")
    if prediction < 0.5:
        print("Dog")
    else:
        print("Not Dog")

    print("Actual: ")
    if y_dog[i] == 0:
        print("Dog")
    else:
        print("Not Dog")
    i += 1
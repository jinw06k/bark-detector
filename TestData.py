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
X_dog, y_dog = load_data("data.json")

i = 2000
wrong = 0
len = X_dog.shape[0]
print(len)
for x in X_dog[2000:3000]:
    prediction = model.predict(x[np.newaxis, ...])

    if prediction[0][0] > 0.5 and y_dog[i] == 0:
        wrong += 1
    i += 1

print("Wrong predictions: ", wrong)
print("Percentage of wrong predictions: ", wrong / i * 100)
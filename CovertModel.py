import tensorflow as tf

model = tf.keras.models.load_model('model.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open('model.tflite', 'wb').write(tflite_model)
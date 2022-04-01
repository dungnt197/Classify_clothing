import tensorflow as tf
from tensorflow import keras

# print(tf.__version__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

labels = {0: "T-shirt/top", 1: "Trouser", 2 : "Pullover", 3:"Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
model = tf.keras.models.load_model("plGiay20e.h5")
y_predicted = model.predict(x_test)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: %.4f'% score[0])
print('Test accuracy %.4f'% score[1])

x_inference = x_test[850]

y_predicted = model.predict(np.array([x_inference]))
predicted_label = np.argmax(y_predicted)
print(labels[predicted_label])


plt.figure()
plt.imshow(x_inference)
plt.show()

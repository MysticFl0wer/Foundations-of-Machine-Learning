from keras.models import Sequential
import keras.utils
import matplotlib.pyplot as plt
import keras
from keras import layers
import pathlib
import numpy as np
import tensorflow as tf

"""
The dataset can be found at
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

batch_size = 32
img_height = 180
img_width = 180
image_size = (img_width, img_height)
epochs = 25

archive = r"archive\chest_xray\chest_xray\train" #the folder where all the testing images are
data_dir = pathlib.Path(archive).with_suffix('')

train_ds = keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

print("Training images complete.")

val_ds = keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

print("Testing images complete")

class_names = train_ds.class_names

data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width,3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(2)
  ])

model.compile(optimizer='adam',
              loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

x_path = r"C:\Users\0753780\Downloads\archive\chest_xray\chest_xray\test\NORMAL\NORMAL2-IM-0373-0001.jpeg" #testing image
img = keras.utils.load_img(x_path, target_size=image_size)
plt.imshow(img)
plt.show()

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
result = class_names[np.argmax(score)]
print(
    "This image most likely shows {} with a {:.2f} percent confidence."
    .format(result, 100 * np.max(score))
)

plt.title(f"{result} - {np.max(score):.2f}%")
plt.imshow(img)
plt.show()
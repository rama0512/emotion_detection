import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

NUM_CLASSES = 7

base = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.4)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(base.input, out)

model.save("emotion_efficientnet.h5")
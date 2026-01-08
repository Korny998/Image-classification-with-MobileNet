from keras.applications import MobileNet
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.models import Model

from constants import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES


def model_maker() -> Model:
    """
    Creates a Keras model based on MobileNet with a custom classification head.
    """
    base_model = MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    custom_model = base_model(inputs)

    custom_model = GlobalAveragePooling2D()(custom_model)

    custom_model = Dense(64, activation='relu')(custom_model)

    custom_model = Dropout(0.5)(custom_model)

    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)

    return Model(inputs=inputs, outputs=predictions)

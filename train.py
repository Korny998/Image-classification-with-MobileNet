from keras.optimizers import Adam

from dataset import get_generators, prepare_dataset
from model import model_maker


def train():
    """Creates and trains the model using the prepared dataset."""
    model = model_maker()

    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    train_generator, validation_generator = get_generators()

    history = model.fit(
        train_generator,
        epochs=12,
        validation_data=validation_generator
    )

    return history


if __name__ == '__main__':
    prepare_dataset()
    train()
    # show_batch(validation_generator[0])
    # # Uncomment to visualize a batch of validation images

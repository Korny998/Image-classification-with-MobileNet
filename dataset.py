import io
import os
import shutil
import zipfile

import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from constants import (
    BASE_DIR, BATCH_SIZE, CLASS_LIST,
    IMAGE_PATH, IMG_HEIGHT, IMG_WIDTH,
    PROJECT_DIR
)


url: str = 'https://storage.yandexcloud.net/academy.ai/cat-and-dog.zip'

request = requests.get(url, timeout=30)
request.raise_for_status()

data_dir: str = os.path.join(PROJECT_DIR, 'data')
os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(os.path.join(data_dir, 'training_set')):
    with zipfile.ZipFile(io.BytesIO(request.content)) as z:
        z.extractall(data_dir)

os.makedirs(BASE_DIR, exist_ok=True)

train_dir, validation_dir, test_dir = [
    os.path.join(BASE_DIR, name) for name in ('train', 'validation', 'test')
]

for folder in (train_dir, validation_dir, test_dir):
    os.makedirs(folder, exist_ok=True)


def create_dataset(
        img_path: str, new_path: str,
        class_name: str, start_index: int,
        end_index: int
) -> None:
    """Copy a subset of images for a given class to a new folder."""
    src_path: str = os.path.join(img_path, class_name)
    dst_path: str = os.path.join(new_path, class_name)

    if not os.path.exists(src_path):
        raise FileNotFoundError(f'Class folder not found: {src_path}')

    os.makedirs(dst_path, exist_ok=True)
    class_files: list[str] = sorted(os.listdir(src_path))
    end_index = min(end_index, len(class_files))

    for file_name in class_files[start_index:end_index]:
        shutil.copyfile(
            os.path.join(src_path, file_name),
            os.path.join(dst_path, file_name)
        )


def clean_directory(path: str) -> None:
    """Cleans a directory by deleting all files and subdirectories."""
    if not os.path.exists(path):
        return

    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)


def prepare_dataset() -> None:
    """
    Prepares train, validation, and test directories:
    - cleans old data;
    - copies images into the respective directories.
    """
    for path in (train_dir, validation_dir, test_dir):
        os.makedirs(path, exist_ok=True)
        clean_directory(path)

    for class_name in CLASS_LIST:
        create_dataset(IMAGE_PATH, train_dir, class_name, 0, 2800)
        create_dataset(IMAGE_PATH, validation_dir, class_name, 2800, 3400)
        create_dataset(IMAGE_PATH, test_dir, class_name, 3400, 4000)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1. / 255)


def get_generators():
    """Creates data generators for training and validation datasets."""
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_generator, validation_generator

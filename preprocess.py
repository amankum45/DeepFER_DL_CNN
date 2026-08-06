"""
preprocess.py
-------------------------------------
Dataset Loading & Preprocessing Module
Project: DeepFER - Facial Emotion Recognition
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataPreprocessor:
    """
    Creates training, validation and test data generators.
    """

    def __init__(
        self,
        train_dir,
        test_dir,
        image_size=(48, 48),
        batch_size=32,
        validation_split=0.2
    ):

        self.train_dir = train_dir
        self.test_dir = test_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split

    def get_data_generators(self):

        # ==========================
        # Training Data Generator
        # ==========================

        train_datagen = ImageDataGenerator(

            rescale=1.0 / 255,

            rotation_range=20,
            width_shift_range=0.20,
            height_shift_range=0.20,
            zoom_range=0.20,
            shear_range=0.15,
            horizontal_flip=True,

            brightness_range=[0.8, 1.2],

            fill_mode="nearest",

            validation_split=self.validation_split
        )

        # ==========================
        # Test Data Generator
        # ==========================

        test_datagen = ImageDataGenerator(
            rescale=1.0 / 255
        )

        # ==========================
        # Train Generator
        # ==========================

        train_generator = train_datagen.flow_from_directory(

            directory=self.train_dir,

            target_size=self.image_size,

            color_mode="grayscale",

            batch_size=self.batch_size,

            class_mode="categorical",

            subset="training",

            shuffle=True
        )

        # ==========================
        # Validation Generator
        # ==========================

        validation_generator = train_datagen.flow_from_directory(

            directory=self.train_dir,

            target_size=self.image_size,

            color_mode="grayscale",

            batch_size=self.batch_size,

            class_mode="categorical",

            subset="validation",

            shuffle=False
        )

        # ==========================
        # Test Generator
        # ==========================

        test_generator = test_datagen.flow_from_directory(

            directory=self.test_dir,

            target_size=self.image_size,

            color_mode="grayscale",

            batch_size=self.batch_size,

            class_mode="categorical",

            shuffle=False
        )

        return train_generator, validation_generator, test_generator
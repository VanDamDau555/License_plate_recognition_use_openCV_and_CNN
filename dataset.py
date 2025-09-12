import tensorflow as tf
import os

class LandD_Dataset:
    def __init__(self, data_dir, inp_shape=(64, 64, 1), batch_size=32, test_split=0.2, seed=123):
        self.data_dir = data_dir
        self.inp_shape = inp_shape
        self.batch_size = batch_size
        self.test_split = test_split
        self.seed = seed
        self.class_names = None

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {self.data_dir}")

    def get_datasets(self):
        # Load train dataset (RGB)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.test_split,
            subset="training",
            seed=self.seed,
            image_size=(self.inp_shape[0], self.inp_shape[1]),
            batch_size=self.batch_size
        )

        # Load test dataset (RGB)
        test_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.test_split,
            subset="validation",
            seed=self.seed,
            image_size=(self.inp_shape[0], self.inp_shape[1]),
            batch_size=self.batch_size
        )
        self.class_names = train_ds.class_names

        # Convert RGB -> Grayscale if [2] 
        if self.inp_shape[2] == 1:
            train_ds = train_ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
            test_ds = test_ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))

        # Normalize to [0,1]
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

        return train_ds, test_ds

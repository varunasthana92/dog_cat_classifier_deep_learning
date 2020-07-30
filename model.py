import tensorflow as tf


class Model():
    def __init__(self, img_size = 150):
        self.model = None
        self.img_size = img_size
        self.gen_model()

    def gen_model(self):
        self.model = tf.keras.models.Sequential([
                        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
                        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding= 'valid',
                                               input_shape=(self.img_size, self.img_size, 3)),
                        tf.keras.layers.MaxPooling2D(2, 2),
                        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding= 'valid'),
                        tf.keras.layers.MaxPooling2D(2, 2),
                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                        tf.keras.layers.MaxPooling2D(2, 2),
                        # Flatten the results to feed into a DNN
                        tf.keras.layers.Flatten(),
                        # 512 neuron hidden layer
                        tf.keras.layers.Dense(512, activation='relu'),
                        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats')
                        # and 1 for the other ('dogs')
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
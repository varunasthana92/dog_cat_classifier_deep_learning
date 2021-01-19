import tensorflow as tf
from dataFiles import *
import tensorflow.keras as tfk
from model import Model
from model import Model2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def pre_process(img_size, batch_size, validation_dir):
    # ImageDataGenerator class allows you to instantiate generators
    # of augmented image batches ( and their labels) via.flow(data, labels)
    # or.flow_from_directory(directory)

    # All images will be rescaled by 1./255.
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    # --------------------
    # Flow validation images in batches using valid_datagen generator
    # --------------------
    validation_generator = valid_datagen.flow_from_directory(validation_dir,
                                                            batch_size=batch_size,
                                                            class_mode='binary',
                                                            color_mode = 'grayscale',
                                                            target_size=(img_size, img_size))
    return validation_generator


def test(model, lr, img_size, batch_size, validation_dir, CheckPointPath):
    model.compile(optimizer= RMSprop(lr= lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    LatestModel = tf.train.latest_checkpoint(CheckPointPath)
    model.load_weights(LatestModel)
    # tfk.utils.plot_model(model, to_file='Model.png', show_shapes=True)
    test_generator = pre_process(img_size, batch_size, validation_dir)
    history = model.evaluate_generator( test_generator, steps=None, callbacks=None, max_queue_size=10, workers=1,
        use_multiprocessing=False, verbose=2)



def main():
    # is_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    # if is_gpu:
    #     tf.debugging.set_log_device_placement(True)

    base_dir = '../Data/cats_and_dogs_filtered'
    CheckPointPath = './checkpoints/'
    # CheckPointPath = CheckPointPath + '{epoch:04d}model.ckpt'
    _, validation_dir, _, _, validation_cats_dir, validation_dogs_dir, _, _ = data_gen(base_dir)
    # show_imgs(train_cat_fnames, train_dog_fnames, train_cats_dir, train_dogs_dir, 10)

    img_size = 150
    batch_size = 20
    lr = 0.0001
    model_obj = Model2(img_size)
    model = model_obj.model
    print(model.summary())
    test(model,lr, img_size, batch_size, validation_dir, CheckPointPath)


if __name__ == '__main__':
    main()
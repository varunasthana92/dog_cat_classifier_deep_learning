import tensorflow as tf
from dataFiles import *
import tensorflow.keras as tfk
from model import Model
from model import Model2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def pre_process(img_size, batch_size, train_dir, validation_dir):
    # ImageDataGenerator class allows you to instantiate generators
    # of augmented image batches ( and their labels) via.flow(data, labels)
    # or.flow_from_directory(directory)

    # All images will be rescaled by 1./255.
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    # --------------------
    # Flow training images in batches using train_datagen generator
    # --------------------
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        color_mode = 'grayscale', #can be 'rgb'
                                                        target_size=(img_size, img_size))
    # --------------------
    # Flow validation images in batches using valid_datagen generator
    # --------------------
    validation_generator = valid_datagen.flow_from_directory(validation_dir,
                                                            batch_size=batch_size,
                                                            class_mode='binary',
                                                            color_mode = 'grayscale',
                                                            target_size=(img_size, img_size))
    return train_generator, validation_generator


def train(model, lr, img_size, batch_size, num_epochs, train_dir, validation_dir, train_data_size, CheckPointPath, LogsPath):
    # NOTE: In this case, using the RMSprop optimization algorithm is preferable to
    # stochastic gradient descent (SGD), because RMSprop automates learning-rate tuning
    # for us. (Other optimizers, such as Adam and Adagrad, also automatically adapt the
    # learning rate during training, and would work equally well here.)
    model.compile(optimizer= RMSprop(lr= lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # we will pre-process the image to normalize it to [0,1] which
    # originally is in [0,255] range. And then we wll resize it to
    # a defined square size
    train_generator, validation_generator = pre_process(img_size, batch_size, train_dir, validation_dir)

    # Create a callback that saves the model's weights every epoch
    CheckPointCallback = tfk.callbacks.ModelCheckpoint(filepath=CheckPointPath, verbose=1, save_weights_only=True, save_freq='epoch')

    # TensorBoard Callback
    TensorBoardCallback = tfk.callbacks.TensorBoard(log_dir=LogsPath, histogram_freq=1, write_graph=True, update_freq=1, profile_batch=2, embeddings_freq=1)

    model.save_weights(CheckPointPath.format(epoch=0))
    # with verbose = 2 we will see 4 values per epoch -- Loss, Accuracy, Validation Loss and Validation Accuracy
    # but training progress bar will not be visible
    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  steps_per_epoch = train_data_size / batch_size,
                                  epochs= num_epochs,
                                  validation_steps= 50,
                                  callbacks=[CheckPointCallback, TensorBoardCallback])
                                  # verbose=2)

def main():
    # is_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    # if is_gpu:
    #     tf.debugging.set_log_device_placement(True)

    base_dir = '../Data/cats_and_dogs_filtered'
    CheckPointPath = './checkpoints/'
    LogsPath = './Logs/'
    CheckPointPath = CheckPointPath + '{epoch:04d}model.ckpt'
    train_dir, validation_dir, train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir, train_cat_fnames, train_dog_fnames = data_gen(base_dir)
    # show_imgs(train_cat_fnames, train_dog_fnames, train_cats_dir, train_dogs_dir, 10)

    img_size = 150
    batch_size = 20
    num_epochs = 15
    lr = 0.0001
    model_obj = Model2(img_size)
    model = model_obj.model
    print(model.summary())
    train(model, lr, img_size, batch_size, num_epochs, train_dir, validation_dir,
             len(train_cat_fnames) + len(train_dog_fnames), CheckPointPath, LogsPath)


if __name__ == '__main__':
    main()
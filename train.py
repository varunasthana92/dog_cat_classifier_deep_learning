import tensorflow as tf
from dataFiles import *
from model import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


def pre_process(img_size, batch_size, train_dir, validation_dir):
    # ImageDataGenerator class allows you to instantiate generators
    # of augmented image batches ( and their labels) via.flow(data, labels)
    # or.flow_from_directory(directory)

    # All images will be rescaled by 1./255.
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    # --------------------
    # Flow training images in batches of 20 using train_datagen generator
    # --------------------
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        target_size=(img_size, img_size))
    # --------------------
    # Flow validation images in batches of 20 using test_datagen generator
    # --------------------
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            batch_size=batch_size,
                                                            class_mode='binary',
                                                            target_size=(img_size, img_size))
    return train_generator, validation_generator


def train(model, img_size, batch_size, num_epochs, train_dir, validation_dir, train_data_size):
    # NOTE: In this case, using the RMSprop optimization algorithm is preferable to
    # stochastic gradient descent (SGD), because RMSprop automates learning-rate tuning
    # for us. (Other optimizers, such as Adam and Adagrad, also automatically adapt the
    # learning rate during training, and would work equally well here.)
    model.compile(optimizer= RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # we will pre-process the image to be of the normalize it to [0,1]
    # which originally is in [0,255] range. And then we wll resize it to
    # a defined square size
    train_generator, validation_generator = pre_process(img_size, batch_size, train_dir, validation_dir)

    # with verbose = 2 we will see 4 values per epoch -- Loss, Accuracy, Validation Loss and Validation Accuracy.
    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  steps_per_epoch = train_data_size / batch_size,
                                  epochs= num_epochs,
                                  validation_steps= 50)
                                  # verbose=2)


def main():
    base_dir = '../Data/cats_and_dogs_filtered'
    train_dir, validation_dir, train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir, train_cat_fnames, train_dog_fnames = data_gen(base_dir)
    # show_imgs(train_cat_fnames, train_dog_fnames, train_cats_dir, train_dogs_dir, 10)

    img_size = 150
    batch_size = 20
    num_epochs = 15
    model_obj = Model(img_size)
    model = model_obj.model
    print(model.summary())
    train(model, img_size, batch_size, num_epochs, train_dir, validation_dir, len(train_cat_fnames) + len(train_dog_fnames))


if __name__ == '__main__':
    main()
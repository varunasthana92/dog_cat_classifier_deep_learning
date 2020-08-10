import numpy as np
import tensorflow as tf
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# import zipfile
# wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip  -O /tmp/cats_and_dogs_filtered.zip
# local_zip = '../Data/cats_and_dogs_filtered.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('../Data/Cat_Dog')
# zip_ref.close()


def data_gen(base_dir, idx=0):
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    # Directory with our training cat/dog pictures
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    # Directory with our validation cat/dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)

    # print(train_cat_fnames[:10])
    # print(len(train_dog_fnames))
    return train_dir, validation_dir, train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir, train_cat_fnames, train_dog_fnames


def show_imgs(train_cat_fnames, train_dog_fnames, train_cats_dir, train_dogs_dir, idx=0):
    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4
    pic_index = idx  # Index for iterating over images
    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)
    pic_index += 8
    next_cat_pix = [os.path.join(train_cats_dir, fname)
                    for fname in train_cat_fnames[pic_index - 8:pic_index]
                    ]
    next_dog_pix = [os.path.join(train_dogs_dir, fname)
                    for fname in train_dog_fnames[pic_index - 8:pic_index]
                    ]
    for i, img_path in enumerate(next_cat_pix + next_dog_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)
        img = mpimg.imread(img_path)
        plt.imshow(img)
    plt.show()
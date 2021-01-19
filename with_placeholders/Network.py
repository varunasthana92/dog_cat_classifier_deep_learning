import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Network():
    def __init__(self, img_size=224):
        # self.model = None
        self.img_size = img_size

    def model(self, input_data):
        layer1 = tf.keras.layers.Conv2D(name='conv1', padding='same', filters=64, kernel_size=3,
                               activation=None, input_shape=(self.img_size, self.img_size, 1))
        net = layer1(input_data)
        net = tf.nn.relu(net, name='relu1')  # could also use activation='relu' in previous layer and avoid this layer
        net = tf.layers.conv2d(inputs=net, name='conv2', padding='same', filters=64, kernel_size=3,
                               activation='relu')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)


        # Block 2
        net = tf.layers.conv2d(inputs=net, name='conv3', padding='same', filters=128, kernel_size=3,
                               activation='relu')
        net = tf.layers.conv2d(inputs=net, name='conv4', padding='same', filters=128, kernel_size=3,
                               activation='relu')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)



        # Block 3
        net = tf.layers.conv2d(inputs=net, name='conv5', padding='same', filters=256, kernel_size=3,
                               activation='relu')
        net = tf.layers.conv2d(inputs=net, name='conv6', padding='same', filters=256, kernel_size=3,
                               activation='relu')
        net = tf.layers.conv2d(inputs=net, name='conv7', padding='same', filters=256, kernel_size=3,
                               activation='relu')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)


        # Block 4
        net = tf.layers.conv2d(inputs=net, name='conv8', padding='same', filters=512, kernel_size=3,
                               activation='relu')
        net = tf.layers.conv2d(inputs=net, name='conv9', padding='same', filters=512, kernel_size=3,
                               activation='relu')
        net = tf.layers.conv2d(inputs=net, name='conv10', padding='same', filters=512, kernel_size=3,
                               activation='relu')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)


        # Block 5
        net = tf.layers.conv2d(inputs=net, name='conv11', padding='same', filters=512, kernel_size=3,
                               activation='relu')
        net = tf.layers.conv2d(inputs=net, name='conv12', padding='same', filters=512, kernel_size=3,
                               activation='relu')
        net = tf.layers.conv2d(inputs=net, name='conv13', padding='same', filters=512, kernel_size=3,
                               activation='relu')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)


        # net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name ='bn1')

        net = tf.layers.flatten(net)

        # Define the Neural Network's fully connected layers:
        # 2 ways of using dense layers
        net = tf.layers.dense(inputs=net, name='fc1', units=512, activation=tf.nn.relu)
        # net = tf.layers.dense(inputs = net, name='fc2', units = 2, activation = 'sigmoid')

        denseLayer = tf.keras.layers.Dense(1, name='fc2', activation='sigmoid')
        net = denseLayer(net)
        prLogits = net
        # prSoftMax is defined as normalized probabilities of the output of the neural network
        prSoftMax = tf.nn.softmax(logits = prLogits)
        return prLogits, prSoftMax


class Network2():
    def __init__(self, img_size=224):
        # self.model = None
        self.img_size = img_size

    def model(self, input_data):
        layer1 = tf.keras.layers.Conv2D(name='conv1', padding='same', filters=64, kernel_size=3,
                               activation=None, input_shape=(self.img_size, self.img_size, 1))
        net = layer1(input_data)
        net = tf.nn.relu(net, name='relu1')  # could also use activation='relu' in previous layer and avoid this layer
        net = tf.layers.conv2d(inputs=net, name='conv2', padding='same', filters=64, kernel_size=3,
                               activation='relu')


        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

        net = tf.layers.conv2d(inputs=net, name='conv2', padding='valid', filters=32, kernel_size=3,
                               activation='relu')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

        net = tf.layers.conv2d(inputs=net, name='conv3', padding='valid', filters=64, kernel_size=3,
                               activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

        # net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name ='bn1')

        net = tf.layers.flatten(net)

        # Define the Neural Network's fully connected layers:
        # 2 ways of using dense layers
        net = tf.layers.dense(inputs=net, name='fc1', units=512, activation=tf.nn.relu)
        # net = tf.layers.dense(inputs = net, name='fc2', units = 1, activation = 'sigmoid')

        denseLayer = tf.keras.layers.Dense(2, name='fc2', activation='sigmoid')
        net = denseLayer(net)
        prLogits = net
        # prSoftMax is defined as normalized probabilities of the output of the neural network
        prSoftMax = tf.nn.softmax(logits = prLogits)
        return prLogits, prSoftMax

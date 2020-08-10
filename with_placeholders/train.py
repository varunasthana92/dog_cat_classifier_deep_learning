import tensorflow as tf
from Network import Network
import argparse
import sys
import os
import glob
import numpy as np
from tqdm import tqdm
import random
try:
    import cv2
except:
    sys.path.remove(sys.path[2])
    import cv2
from dataUtils import *


def ReadLabels(LabelsPathTrain):
    if(not (os.path.isfile(LabelsPathTrain))):
        print('ERROR: Train Labels do not exist in '+LabelsPathTrain)
        sys.exit()
    else:
        TrainLabels = open(LabelsPathTrain, 'r')
        TrainLabels = TrainLabels.read()
        TrainLabels = TrainLabels.split('\n')
        TrainLabels = list(np.array(TrainLabels).astype(np.int))
    return TrainLabels


def ReadDirNames(ReadPath):
    if not (os.path.isfile(ReadPath)):
        print('ERROR: Train Labels do not exist in '+ ReadPath)
        sys.exit()
    else:
        DirNames = open(ReadPath, 'r')
        DirNames = DirNames.read()
        DirNames = DirNames.split('\n')
    return DirNames


def FindLatestModel(CheckPointPath):
    """
    Finds Latest Model in CheckPointPath
    Inputs:
    CheckPointPath - Path where you have stored checkpoints
    Outputs:
    LatestFile - File Name of the latest checkpoint
    """
    FileList = glob.glob(CheckPointPath + '*.ckpt.index') # * means all if need specific format then *.csv
    LatestFile = max(FileList, key=os.path.getctime)
    # Strip everything else except needed information
    LatestFile = LatestFile.replace(CheckPointPath, '')
    LatestFile = LatestFile.replace('.ckpt.index', '')
    return LatestFile


def GenerateBatch(BasePath, ImageSize, BatchSize, File_Names, File_Labels):
    ImgBatch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < BatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(File_Names))
        
        RandImageName = BasePath + os.sep + File_Names[RandIdx] 
        ImageNum += 1

        img = cv2.imread(RandImageName,0) # read in GrayScale mode
        img = cv2.resize(img, (ImageSize, ImageSize))
        img = img.reshape((ImageSize, ImageSize, 1))
        img = np.float32(img*1./255)

        Label = File_Labels[RandIdx]
        ImgBatch.append(img)
        LabelBatch.append(Label)
        
    return ImgBatch, LabelBatch


def train(ImgPH, LabelPH, lr, img_size, MiniBatchSize, NumEpochs, BasePath, DirNamesTrain, TrainLabels,
          CheckPointPath, LatestFile, LogsPath, DivTrain):
    TotalImgs = len(DirNamesTrain)
    Train_Names = DirNamesTrain[:int(TotalImgs*0.8)]
    Train_Labels = TrainLabels[:int(TotalImgs*0.8)]
    
    Valid_Names = DirNamesTrain[int(TotalImgs*0.2):]
    Valid_Labels = TrainLabels[int(TotalImgs*0.2):]

    model_obj = Network(img_size)
    print(model_obj)
    prLogits, prSoftMax = model_obj.model(ImgPH)

    with tf.name_scope('Loss'):
        ###############################################
        """
        diff between LOSS functions: sparse_softmax_cross_entropy_with_logits and softmax_cross_entropy_with_logits
        https://stackoverflow.com/questions/37312421/whats-the-difference-between-sparse-softmax-cross-entropy-with-logits-and-softm
        """
        ###############################################
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = LabelPH, logits = prLogits)
        loss = tf.reduce_mean(cross_entropy)


    with tf.name_scope('Accuracy'):
        """
        shape of prLogits = [batch_size]
        shape of LabelPH = [batch_size]
        """
        prSoftMaxDecoded = tf.argmax(prSoftMax, axis=1)
        Acc = tf.reduce_mean(tf.cast(tf.math.equal(prSoftMaxDecoded, tf.cast(LabelPH, dtype = tf.int64)), dtype=tf.float32))

        #################################################
        """
        if one-hot encoded labels are used with softmax
        shape of prSoftMax = [batch_size, numClasses]
        shape of LabelPH = [batch_size, numClasses]
        """

        # prSoftMaxDecoded = tf.argmax(prSoftMax, axis=1)
        # LabelDecoded = tf.argmax(LabelPH, axis=1)
        # Acc = tf.reduce_mean(tf.cast(tf.math.equal(prSoftMaxDecoded, LabelDecoded), dtype=tf.float32))
        #################################################

    with tf.name_scope('RMSProp'):
        Optimizer = tf.train.RMSPropOptimizer(learning_rate = lr).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    TrainLossSummary = tf.summary.scalar('TrainLoss', loss)
    ValidLossSummary = tf.summary.scalar('ValidLoss', loss)

    TrainAccuracySummary = tf.summary.scalar('TrainAccuracy', Acc)
    ValidAccuracySummary = tf.summary.scalar('ValidAccuracy', Acc)
    # Merge all summaries into a single operation
    TrainingSummary = tf.summary.merge([TrainLossSummary, TrainAccuracySummary])
    ValidSummary = tf.summary.merge([ValidLossSummary, ValidAccuracySummary])

    # Setup Saver
    Saver = tf.train.Saver(max_to_keep=None)

    acc_t= []
    loss_t = []
    acc_v = []
    loss_v = []

    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
        
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(len(Train_Names)/MiniBatchSize/DivTrain)
            ### For training
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                ImgBatch, LabelBatch = GenerateBatch(BasePath, img_size, MiniBatchSize, Train_Names, Train_Labels)
                FeedDict = {ImgPH: ImgBatch, LabelPH: LabelBatch}
                _, LossThisBatch, TSummary, AccThisBatch = sess.run([Optimizer, loss, TrainingSummary, Acc], feed_dict=FeedDict)
                loss_t.append(LossThisBatch)
                acc_t.append(AccThisBatch)
                if PerEpochCounter % 10 == 0:
                    print("Accuracy of model : " + str(sum(acc_t) / len(acc_t)))
                    print("Loss of model : " + str(sum(loss_t)))
                # Tensorboard
                Writer.add_summary(TSummary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                Writer.flush()

            NumIterationsPerEpoch = int(len(Valid_Names)/MiniBatchSize/DivTrain)
            ### For training
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                ImgBatch, LabelBatch = GenerateBatch(BasePath, img_size, MiniBatchSize, Valid_Names, Valid_Labels)
                FeedDict = {ImgPH: ImgBatch, LabelPH: LabelBatch}
                _, LossThisBatch, VSummary, AccThisBatch = sess.run([Optimizer, loss, ValidSummary, Acc], feed_dict=FeedDict)
                loss_v.append(LossThisBatch)
                acc_v.append(AccThisBatch)
                # Tensorboard
                Writer.add_summary(VSummary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                Writer.flush()
            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved... ')
            print("After epoch")
            
            print("Net Accuracy of model : " + str(sum(acc_t)/len(acc_t)))
            print("Total Loss of model : " + str(sum(loss_t)))
            print("Net validation accuracy : " + str(sum(acc_v)/len(acc_v)))
            print("Net validation loss : " + str(sum(loss_v)))
            acc_t, loss_t, acc_v, loss_v = [],[],[],[]


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='../Data/dogs-vs-cats/', help='Base path of images')
    Parser.add_argument('--CheckPointPath', default='./Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=1, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=50, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='./Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    Parser.add_argument('--ImageSize', type=int, default=150)
    Parser.add_argument('--lr', type=float, default=0.001)
    Args = Parser.parse_args()
    

    # train_dir, validation_dir, train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir, train_cat_fnames, train_dog_fnames = data_gen(base_dir)
    # show_imgs(train_cat_fnames, train_dog_fnames, train_cats_dir, train_dogs_dir, 10)


    BasePath =Args.BasePath + 'train/'
    img_size = Args.ImageSize
    MiniBatchSize = Args.MiniBatchSize
    NumEpochs = Args.NumEpochs
    lr = Args.lr
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    DivTrain = float(Args.DivTrain)

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    DirNamesTrain = ReadDirNames('./trainFileNames.txt')
    TrainLabels = ReadLabels('./TrainLabel.txt' )

    # print(model.summary())
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, img_size, img_size, 1))
    LabelPH = tf.placeholder(tf.int32, shape=(MiniBatchSize)) # OneHOT labels not used
    train(ImgPH, LabelPH, lr, img_size, MiniBatchSize, NumEpochs, BasePath, DirNamesTrain, TrainLabels,
          CheckPointPath, LatestFile, LogsPath, DivTrain)


if __name__ == '__main__':
    main()
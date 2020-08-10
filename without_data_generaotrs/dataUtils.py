import os
try:
    import cv2
except:
    sys.path.remove(sys.path[2])
    import cv2
import numpy as np

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
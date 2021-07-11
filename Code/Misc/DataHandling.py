import Misc.ImageUtils as iu
import os
import numpy as np
import glob

def SetupAll(Args):
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    DirNames - Full path to all image files without extension
    Train/Val/Test - Idxs of all the images to be used for training/validation (held-out testing in this case)/testing
    Ratios - Ratios is a list of fraction of data used for [Train, Val, Test]
    CheckPointPath - Path to save checkpoints/model
    OptimizerParams - List of all OptimizerParams: depends on Optimizer
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrain/Val/TestSamples - length(Train/Val/Test)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    Train/Val/TestLabels - Labels corresponding to Train/Val/Test
    """
    # Setup DirNames
    DirNames = glob.glob(Args.BasePath+'/Imgs/*.'+Args.ImgFormat)
    TrainNames = DirNames # Use entire dataset for training

    # Setup Neural Net Params
    # List of all OptimizerParams: depends on Optimizer
    # For ADAM Optimizer: [LearningRate, Beta1, Beta2, Epsilion]
    UseDefaultFlag = 0 # Set to 0 to use your own params, do not change default parameters
    if UseDefaultFlag:
        # Default Parameters
        OptimizerParams = [1e-3, 0.9, 0.999, 1e-8]
    else:
        # Custom Parameters
        OptimizerParams = [Args.LR, 0.9, 0.999, 1e-8]   
        
    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 1000 
    # Number of passes of Val data with MiniBatchSize 
    NumTestRunsPerEpoch = 5
    
    # Image Input Shape
    ImageSize = np.array([480, 480, 1])
    NumTrainSamples = len(TrainNames)

    # Pack everything into args
    Args.TrainNames = TrainNames
    Args.OptimizerParams = OptimizerParams
    Args.SaveCheckPoint = SaveCheckPoint
    Args.NumTrainSamples = NumTrainSamples
    Args.ImageSize = ImageSize
    Args.NumOut = 1 # Grayscale Image
    
    return Args

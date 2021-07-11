#!/usr/bin/env python3

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (sudo)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)

import tensorflow as tf
import cv2
import sys
import os
import glob
import re
import Misc.ImageUtils as iu
import random
import matplotlib.pyplot as plt
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import Misc.TFUtils as tu
from Misc.DataHandling import *
from Misc.BatchCreationTestSingleTF import *
from Misc.Decorators import *
# Import of network is done in main code
import importlib
from datetime import datetime
import getpass
import copy
import platform

# Don't generate pyc codes
sys.dont_write_bytecode = True


def PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN):
    # TODO: Write to file?
    Username = getpass.getuser()
    cprint('Running on {}'.format(Username), 'yellow')
    cprint('Network Statistics', 'yellow')
    cprint('Network Used: {}'.format(Args.NetworkName), 'yellow')
    cprint('GPU Used: {}'.format(Args.GPUDevice), 'yellow')
    cprint('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                          VN.NumBlocks, VN.NumSubBlocks, VN.DropOutRate), 'yellow')
    cprint('Num Params: {}'.format(NumParams), 'green')
    cprint('Num FLOPs: {}'.format(NumFlops), 'green')
    cprint('Estimated Model Size (MB): {}'.format(ModelSize), 'green')
    cprint('Model loaded from: {}'.format(Args.CheckPointPath), 'red')
        
def TestOperation(InputPH, Args):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    HomingVecPH is the ground truth  homing vector placeholder
    DirNames - Full path to all image files without extension
    Train/Val - Idxs of all the images to be used for training/validation (held-out testing in this case)
    Train/ValLabels - Labels corresponding to Train/Val
    NumTrain/ValSamples - length(Train/Val)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    OptimizerParams - List of all OptimizerParams: depends on Optimizer
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of data
    LatestFile - Latest checkpointfile to continue training
    Outputs:
    Saves Trained network in CheckPointPath
    """
    # Create Network Object with required parameters
    ClassName = Args.NetworkName.replace('Network.', '').split('Net')[0]+'Net'
    Network = getattr(Args.Net, ClassName)
    VN = Network(InputPH = InputPH, InitNeurons = Args.InitNeurons, Suffix = Args.Suffix)

    # Predict output with forward pass
    prVal = VN.Network()
   
    # Setup Saver
    Saver = tf.train.Saver()

    # File to save errors
    FileName = Args.WritePath + 'Evaluation.txt'
    Logs = open(FileName, 'w')
    Logs.write('FileName, MSE Error\n')


    with tf.Session() as sess:       
        Saver.restore(sess, Args.CheckPointPath)
        # Extract only numbers from the name
        print('Loaded checkpoints ....')

        # Create Batch Generator Object
        bg = BatchGeneration(sess, InputPH)

        # Print out Number of parameters
        NumParams = tu.FindNumParams(1)
        # Print out Number of Flops
        NumFlops = tu.FindNumFlops(sess, 1)
        # Print out Expected Model Size
        ModelSize = tu.CalculateModelSize(1)

        # Pretty Print Stats
        PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN)

        # Predictions File
        ModelName = Args.CheckPointPath.split('/')[-1]

        for count in tqdm(range(len(Args.DirNames))):
            Args.Input = Args.DirNames[count]
            IBatch, LabelBatch = bg.GenerateBatchTF(Args)
            
            FeedDict = {VN.InputPH: IBatch}
            Timer1 = mu.tic()
            prValRet = np.squeeze(sess.run([prVal], feed_dict=FeedDict))
            print('Time(FPS) for last iteration: %f(%f) secs(/sec)'%(mu.toc(Timer1), 1./mu.toc(Timer1)))

            A = np.squeeze(LabelBatch[0])
            B = prValRet
            B = (np.clip(B, 0., 1.)*255.).astype(np.uint8)
            
            Error = np.square(A-B)
            MeanErrorPx = np.mean(Error)
            Logs.write(Args.Input + ','  '{}'.format(MeanErrorPx))
            
            IDisp = np.squeeze(iu.PolImgDisp(IBatch[0]))
            Overlay = iu.OverlayCornersOnImg(IDisp, B)
            IDisp = np.tile(IDisp[:,:,np.newaxis], (1,1,3))
 
            if(Args.Vis):
                print('Mean Square Error: {}'.format(MeanErrorPx))
                cv2.imshow('Img, Overlay', np.hstack((IDisp, Overlay)))
                
                cv2.imshow('GT, Pred', np.hstack((A, B)))
                cv2.waitKey(0)

            WriteName = Args.Input.split('/')[-1]
            WriteName = Args.WritePath + 'Pred/' + WriteName 
            cv2.imwrite(WriteName, B)
            WriteName = WriteName.replace('Pred', 'Overlays')
            cv2.imwrite(WriteName, Overlay)
                
        # Pretty Print Stats before exiting
        PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN)
    
        
        
def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NetworkType', default='Large', help='Choice of Network type, choose from Small, Large, Default:Large')
    Parser.add_argument('--NetworkName', default='Network.ResNet', help='Name of network file, Default: Network.ResNet')
    Parser.add_argument('--CheckPointPath', default='/media/nitin/Education/DVSProp/CheckPoints/SingleStyleProp/49model.ckpt', \
        help='Path to save checkpoints, Default:/media/nitin/Education/DVSProp/CheckPoints/SingleStyleProp/49model.ckpt')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--InitNeurons', type=float, default=32, help='Number of starting neurons, Default: 32')
    Parser.add_argument('--Suffix', default='', help='Suffix for Naming Network, Default: ''')
    Parser.add_argument('--Input', default='None', help='Image Path should include extension if using single mode else doesnt, Default: None')
    Parser.add_argument('--GT', default='None', help='Ground Truth Path, Default: None')
    Parser.add_argument('--TestMode', default='S', help='S: Single, M: Multiple, Default: S')
    Parser.add_argument('--ImgFormat', default='png', help='Image extension only used for M mode, Default: png')
    Parser.add_argument('--WritePath', default='/media/nitin/Education/DVSProp/Outputs/', help='Path to save results, Default: /media/nitin/Education/DVSProp/Outputs/')
    Parser.add_argument('--Vis', type=int, default=0, help='0 for disabling and 1 for enabling viusalization of outputs, Default: 0')
    Parser.add_argument('--ImgSize', default='[480, 480, 1]', help='Image Size as a list, Default: [480, 480, 1]')
    

    Args = Parser.parse_args()
    
    # Import Network Module
    Args.Net = importlib.import_module(Args.NetworkName)

    # Set GPUDevice
    tu.SetGPU(Args.GPUDevice)

    # Setup all needed parameters including file reading
    Args.ImgSize = np.array(Args.ImgSize.strip('[]').split(',')).astype(int)
    
    # Number of Output channels
    Args.NumOut = 1

    # Define PlaceHolder variables for Input and Predicted output
    InputPH = tf.placeholder(tf.float32, shape=(1, Args.ImgSize[0], Args.ImgSize[1], Args.ImgSize[2]), name='Input')

    # If WritePath doesn't exist make the path
    if(not ((os.path.isdir(Args.WritePath + '/Pred')) or (os.path.isdir(Args.WritePath + '/Overlays')))):
       os.makedirs(Args.WritePath + '/Pred')
       os.makedirs(Args.WritePath + '/Overlays')

    # Get all image names if multiple test mode
    if(Args.TestMode == 'M'):
        Args.DirNames = glob.glob(Args.Input + '/*.' + Args.ImgFormat)
    else:
        Args.DirNames = [Args.Input]
        
    TestOperation(InputPH, Args)
    
    
if __name__ == '__main__':
    main()


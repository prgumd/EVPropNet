#!/usr/bin/env python3

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (sudo)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)


# TODO: Adapt more augmentation from: https://github.com/sthalles/deeplab_v3/blob/master/preprocessing/inception_preprocessing.py

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
from Misc.BatchCreationTF import *
from Misc.Decorators import *
# Import of network is done in main code
import importlib
from datetime import datetime
import getpass
import copy
import platform

# Don't generate pyc codes
sys.dont_write_bytecode = True

@Scope
def Loss(LabelPH, prVal, Args):
    prProb = prVal[:,:,:,:Args.NumOut]
    # Supervised L2 loss
    lossPhoto = tf.reduce_mean(tf.square(prProb - LabelPH))

    return lossPhoto


@Scope
def Optimizer(OptimizerParams, loss):
    Optimizer = tf.train.AdamOptimizer(learning_rate=OptimizerParams[0], beta1=OptimizerParams[1],
                                           beta2=OptimizerParams[2], epsilon=OptimizerParams[3])
    Gradients = Optimizer.compute_gradients(loss)
    OptimizerUpdate = Optimizer.apply_gradients(Gradients)
    # Optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-8).minimize(loss)
    # Optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True).minimize(loss)
    return OptimizerUpdate

def TensorBoard(loss, IPH, prVal, LabelPH, MaskPH, Args):
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    tf.summary.image('I', IPH[:,:,:,0:3], max_outputs=3)
    tf.summary.image('prVal', prVal[:,:,:,0:1], max_outputs=3)
    tf.summary.image('Label', LabelPH[:,:,:,0:1], max_outputs=3)
    tf.summary.image('Mask', MaskPH[:,:,:,0:1], max_outputs=3)
    tf.summary.histogram('LabelHist', LabelPH)
    tf.summary.histogram('prHist', prVal)   
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()
    return MergedSummaryOP


def PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN, OverideKbInput=False):
    # TODO: Write to file?
    Username = getpass.getuser()
    cprint('Running on {}'.format(Username), 'yellow')
    cprint('Network Statistics', 'yellow')
    cprint('Network Used: {}'.format(Args.NetworkName), 'yellow')
    cprint('GPU Used: {}'.format(Args.GPUDevice), 'yellow')
    cprint('Learning Rate: {}'.format(Args.LR), 'yellow')
    cprint('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                          VN.NumBlocks, VN.NumSubBlocks, VN.DropOutRate), 'yellow')
    cprint('Num Params: {}'.format(NumParams), 'green')
    cprint('Num FLOPs: {}'.format(NumFlops), 'green')
    cprint('Estimated Model Size (MB): {}'.format(ModelSize), 'green')
    cprint('CheckPoints are saved in: {}'.format(Args.CheckPointPath), 'red')
    cprint('Logs are saved in: {}'.format(Args.LogsPath), 'red')
    cprint('Images used for Training are in: {}'.format(Args.BasePath), 'red')
    if(OverideKbInput):
        Key = 'y'
    else:
        PythonVer = platform.python_version().split('.')[0]
        # Parse Python Version to handle super accordingly
        if (PythonVer == '2'):
            Key = raw_input('Enter y/Y/yes/Yes/YES to save to RunCommand.md, any other key to exit.')
        else:
            Key = input('Enter y/Y/yes/Yes/YES to save to RunCommand.md, any other key to exit.')
    if(Key.lower() == 'y' or Key.lower() == 'yes'):
        FileName = 'RunCommand.md'
        with open(FileName, 'a+') as RunCommand:
            RunCommand.write('\n\n')
            RunCommand.write('{}\n'.format(datetime.now()))
            RunCommand.write('Username: {}\n'.format(Username))
            RunCommand.write('Learning Rate: {}\n'.format(Args.LR))
            RunCommand.write('Network Used: {}\n'.format(Args.NetworkName))
            RunCommand.write('GPU Used: {}\n'.format(Args.GPUDevice))
            RunCommand.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}\n'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                                              VN.NumBlocks, VN.NumSubBlocks,  VN.DropOutRate))
            RunCommand.write('Num Params: {}\n'.format(NumParams))
            RunCommand.write('Num FLOPs: {}\n'.format(NumFlops))
            RunCommand.write('Estimated Model Size (MB): {}\n'.format(ModelSize))
            RunCommand.write('CheckPoints are saved in: {}\n'.format(Args.CheckPointPath))
            RunCommand.write('Logs are saved in: {}\n'.format(Args.LogsPath))
            RunCommand.write('Images used for Training are in: {}\n'.format(Args.BasePath))
        cprint('Log written in {}'.format(FileName), 'yellow')
        FileName = Args.CheckPointPath + 'RunCommand.md'
        with open(FileName, 'w+') as RunCommand:
            RunCommand.write('\n\n')
            RunCommand.write('{}\n'.format(datetime.now()))
            RunCommand.write('Username: {}\n'.format(Username))
            RunCommand.write('Learning Rate: {}\n'.format(Args.LR))
            RunCommand.write('Network Used: {}\n'.format(Args.NetworkName))
            RunCommand.write('GPU Used: {}\n'.format(Args.GPUDevice))
            RunCommand.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}\n'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                                              VN.NumBlocks, VN.NumSubBlocks, VN.DropOutRate))
            RunCommand.write('Num Params: {}\n'.format(NumParams))
            RunCommand.write('Num FLOPs: {}\n'.format(NumFlops))
            RunCommand.write('Estimated Model Size (MB): {}\n'.format(ModelSize))
            RunCommand.write('CheckPoints are saved in: {}\n'.format(Args.CheckPointPath))
            RunCommand.write('Logs are saved in: {}\n'.format(Args.LogsPath))
            RunCommand.write('Images used for Training are in: {}\n'.format(Args.BasePath))
        cprint('Log written in {}'.format(FileName), 'yellow')
    else:
        cprint('Log writing skipped', 'yellow')
        
    
def TrainOperation(InputPH, IPH, LabelPH, MaskPH, Args):
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
    VN = Network(InputPH = InputPH, InitNeurons = Args.InitNeurons, Suffix = Args.Suffix, NumOut = Args.NumOut)

    # Predict output with forward pass
    # WarpI1Patch contains warp of both I1 and I2, extract first three channels for useful data
    prVal = VN.Network()

    # Compute Loss
    loss = Loss(LabelPH, prVal, Args)

    # Run Backprop and Gradient Update
    OptimizerUpdate = Optimizer(Args.OptimizerParams, loss)
        
    # Tensorboard
    MergedSummaryOP = TensorBoard(loss, IPH, prVal, LabelPH, MaskPH, Args)
 
    # Setup Saver
    Saver = tf.train.Saver()

    try:
        with tf.Session() as sess:       
            if Args.LatestFile is not None:
                Saver.restore(sess, Args.CheckPointPath + Args.LatestFile + '.ckpt')
                # Extract only numbers from the name
                StartEpoch = int(''.join(c for c in Args.LatestFile.split('a')[0] if c.isdigit())) + 1
                print('Loaded latest checkpoint with the name ' + Args.LatestFile + '....')
            else:
                sess.run(tf.global_variables_initializer())
                StartEpoch = 0
                print('New model initialized....')

            # Create Batch Generator Object
            bg = BatchGeneration(sess, IPH)

            # Print out Number of parameters
            NumParams = tu.FindNumParams(1)
            # Print out Number of Flops
            NumFlops = tu.FindNumFlops(sess, 1)
            # Print out Expected Model Size
            ModelSize = tu.CalculateModelSize(1)

            # Pretty Print Stats
            PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN, OverideKbInput=False)

            # Tensorboard
            Writer = tf.summary.FileWriter(Args.LogsPath, graph=tf.get_default_graph())

            for Epochs in tqdm(range(StartEpoch, Args.NumEpochs)):
                NumIterationsPerEpoch = int(Args.NumTrainSamples/Args.MiniBatchSize/Args.DivTrain)
                for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                    IBatch, LabelBatch, MaskBatch, OverlaysBatch = bg.GenerateBatchTF(Args)

                    FeedDict = {VN.InputPH: IBatch, IPH: OverlaysBatch, LabelPH: LabelBatch, MaskPH: MaskBatch}
                    _, LossThisBatch, Summary = sess.run([OptimizerUpdate, loss, MergedSummaryOP], feed_dict=FeedDict)
                   
                    # Tensorboard
                    Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                    # If you don't flush the tensorboard doesn't update until a lot of iterations!
                    Writer.flush()

                    # Save checkpoint every some SaveCheckPoint's iterations
                    if PerEpochCounter % Args.SaveCheckPoint == 0:
                        # Save the Model learnt in this epoch
                        SaveName =  Args.CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                        Saver.save(sess,  save_path=SaveName)
                        print(SaveName + ' Model Saved...')

                # Save model every epoch
                SaveName = Args.CheckPointPath + str(Epochs) + 'model.ckpt'
                Saver.save(sess, save_path=SaveName)
                print(SaveName + ' Model Saved...')

        # Pretty Print Stats before exiting
        PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN, OverideKbInput=True)
    
    except KeyboardInterrupt:
        # Pretty Print Stats before exitting
        PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN, OverideKbInput=False)


        
        
def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/nitin/Datasets/DVSProp', help='Base path of images, Default:/home/nitin/Datasets/DVSProp')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:10')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=32, help='Size of the MiniBatch to use, Default:32')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointPath?, Default:0')
    Parser.add_argument('--RemoveLogs', type=int, default=0, help='Delete log Files from ./Logs?, Default:0')
    Parser.add_argument('--NetworkType', default='Large', help='Choice of Network type, choose from Small, Large, Default:Large')
    Parser.add_argument('--NetworkName', default='Network.ResNet', help='Name of network file, Default: Network.VanillaNet')
    Parser.add_argument('--CheckPointPath', default='/media/nitin/Education/DVSProp/CheckPoints/', help='Path to save checkpoints, Default:/media/nitin/Education/DVSProp/CheckPoints/')
    Parser.add_argument('--LogsPath', default='/media/nitin/Education/DVSProp/Logs/', help='Path to save Logs, Default:/media/nitin/Education/DVSProp/Logs/')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--LR', type=float, default=1e-4, help='Learning Rate, Default: 1e-4')
    Parser.add_argument('--InitNeurons', type=float, default=32, help='Learning Rate, Default: 64')
    Parser.add_argument('--Suffix', default='', help='Suffix for Naming Network, Default: ''')
    Parser.add_argument('--ImgFormat', default='png', help='Image format, Default: png')
    
    
    Args = Parser.parse_args()
    
    # Import Network Module
    Args.Net = importlib.import_module(Args.NetworkName)

    # Set GPUDevice
    tu.SetGPU(Args.GPUDevice)

    if(Args.RemoveLogs is not 0):
        shutil.rmtree(os.getcwd() + os.sep + 'Logs' + os.sep)

    # Setup all needed parameters including file reading
    Args = SetupAll(Args)    

    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(Args.CheckPointPath))):
       os.makedirs(Args.CheckPointPath)
    
    # Find Latest Checkpoint File
    if Args.LoadCheckPoint==1:
        Args.LatestFile = FindLatestModel(Args.CheckPointPath)
    else:
        Args.LatestFile = None
        
    # Define PlaceHolder variables for Input and Predicted output
    InputPH = tf.placeholder(tf.float32, shape=(Args.MiniBatchSize, Args.ImageSize[0], Args.ImageSize[1], Args.ImageSize[2]), name='Input')

    # PH for losses
    IPH = tf.placeholder(tf.float32, shape=(Args.MiniBatchSize, Args.ImageSize[0], Args.ImageSize[1], Args.ImageSize[2]), name='I')
    
    LabelPH =  tf.placeholder(tf.float32, shape=(Args.MiniBatchSize, Args.ImageSize[0], Args.ImageSize[1], Args.ImageSize[2]), name='Label')
    MaskPH =  tf.placeholder(tf.float32, shape=(Args.MiniBatchSize, Args.ImageSize[0], Args.ImageSize[1], Args.ImageSize[2]), name='Mask')
   
    TrainOperation(InputPH, IPH, LabelPH, MaskPH, Args)
    
    
if __name__ == '__main__':
    main()


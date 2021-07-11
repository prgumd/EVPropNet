import random
import os
import cv2
import numpy as np
import tensorflow as tf
import Misc.ImageUtils as iu
import Misc.MiscUtils as mu
import scipy.io as sio
import imageio 

class BatchGeneration():
    def __init__(self, sess, IPH):
        self.sess = sess
        self.IPH = IPH

    def GenerateBatchTF(self, Args):
        """
        Inputs: 
        DirNames - Full path to all image files without extension
        NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
        TrainLabels - Labels corresponding to Train
        NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
        ImageSize - Size of the Image
        MiniBatchSize is the size of the MiniBatch
        Outputs:
        I1Batch - Batch of I1 images after standardization and cropping/resizing to ImageSize
        HomeVecBatch - Batch of Homing Vector labels
        """
        
        IBatch = []
        LabelBatch = []

        I = cv2.imread(Args.Input, 0)
        I = I[:,:,np.newaxis]

        
        if(Args.GT != 'None'):
            try:
                Label = cv2.imread(Args.GT, 0)
                Label = Label[:,:,np.newaxis]
            except:
                Label = np.zeros((np.shape(I)[0], np.shape(I)[1], 1))
        else:
            Label = np.zeros((np.shape(I)[0], np.shape(I)[1], 1))

        if((np.shape(I)[0] >= Args.ImgSize[0]) & (np.shape(I)[1] >= Args.ImgSize[1]) ):
            I = iu.CenterCrop(I, Args.ImgSize)
            Label = iu.CenterCrop(Label, Args.ImgSize)
        else:
            I = cv2.resize(I, tuple(Args.ImgSize[:2]))
            Label = cv2.resize(I, tuple(Args.ImgSize[:2]))
            I = I[:,:,np.newaxis]
            Label = Label[:,:,np.newaxis]
            
        IBatch.append(I)
        LabelBatch.append(Label)
      
        # Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        IBatch = np.sign(np.float32(IBatch)-127.)
        # Label1Batch = iu.StandardizeInputs(np.float32(Label1Batch))

        return IBatch, LabelBatch


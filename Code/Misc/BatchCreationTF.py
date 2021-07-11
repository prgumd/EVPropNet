import random
import os
import cv2
import numpy as np
import tensorflow as tf
import Misc.ImageUtils as iu
import Misc.MiscUtils as mu
import scipy.io as sio
import imageio 
import matplotlib.pyplot as plt


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
        MaskBatch = []
        OverlaysBatch = []
        
        ImageNum = 0
        while ImageNum < Args.MiniBatchSize:
            # Generate random image
            RandIdx = random.randint(0, len(Args.TrainNames)-1)
            RandImageName = Args.TrainNames[RandIdx]

            I = cv2.imread(RandImageName, 0)           
            
            LabelName = RandImageName.replace('Imgs', 'Labels')
            MaskName = RandImageName.replace('Imgs', 'Masks')
            OverlaysName = RandImageName.replace('Imgs', 'Overlays')
            
            if (I is None):
                continue
            
            Label = cv2.imread(LabelName, 0)
            Mask = cv2.imread(MaskName, 0)
            Overlays = cv2.imread(OverlaysName, 0)

            ImageNum += 1
            IBatch.append(I[:,:,np.newaxis])
            LabelBatch.append(Label[:,:,np.newaxis])
            MaskBatch.append(Mask[:,:,np.newaxis])
            OverlaysBatch.append(Overlays[:,:,np.newaxis])            
        
        # Normalize Dataset
        IBatch = np.sign(np.float32(IBatch)-127.)
        LabelBatch = np.float32(LabelBatch)/255.

        return IBatch, LabelBatch, MaskBatch, OverlaysBatch


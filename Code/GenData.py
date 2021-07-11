#!/usr/bin/env python

import cv2
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import imutils
import random
import glob
import argparse
from tqdm import tqdm
import os
from PIL import Image
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation as Rot
import Misc.MiscUtils as mu
import Misc.ImageUtils as iu


class PropGen(object):
    def __init__(self, ImageSize = [1000, 1000], RadiusPx = 400., BladeStartYTopPct = 8.6, BladeStartYBotPct = 8., HubRadiusPct = 14., NumBlades = 2,\
                 PtsXTopPct = [23, 56, 76, 100], PtsXBotPct = [30., 56., 76., 100.], PtsYTopPct = [10.5, 19.6, 14.3, 7.2], PtsYBotPct = [7.9, 11.2, 9.8, 8.0]):
        # ImageSize = [300, 300], RadiusPx = 50., BladeStartYTopPct = 8.6, BladeStartYBotPct = 8., HubRadiusPct = 14., NumBlades = 1,\
        #          PtsXTopPct = [23, 56, 76, 100], PtsXBotPct = [30., 56., 76., 100.], PtsYTopPct = [10.5, 19.6, 14.3, 7.2], PtsYBotPct = [7.9, 11.2, 9.8, 8.0]
        # X is horizontal and Y is vertical
        # All Pct values are Pct of RadiusPx
        self.ImageSize = ImageSize
        self.RadiusPx = RadiusPx
        self.BladeStartYTopPct = BladeStartYTopPct
        self.BladeStartYBotPct = BladeStartYBotPct
        self.HubRadiusPct = HubRadiusPct
        self.NumBlades = NumBlades
        self.PtsXTopPct = np.array(PtsXTopPct)
        self.PtsXBotPct = np.array(PtsXBotPct)
        self.PtsYTopPct = np.array(PtsYTopPct)
        self.PtsYBotPct = -np.array(PtsYBotPct)
        self.Img = np.zeros(self.ImageSize)

    def GenOneProp(self):
        # Initialize with zeros
        self.Img = np.zeros(self.ImageSize)
        # Top part of the blade
        PtsXTopPx = np.insert(self.RadiusPx*self.PtsXTopPct/100. + self.RadiusPx*self.HubRadiusPct/100., 0, self.RadiusPx*self.HubRadiusPct/100.)
        PtsYTopPx = np.insert(self.RadiusPx*self.PtsYTopPct/100., 0, self.RadiusPx*self.BladeStartYTopPct/100.)
        # Bottom part of the blade
        PtsXBotPx = np.insert(self.RadiusPx*self.PtsXBotPct/100. + self.RadiusPx*self.HubRadiusPct/100., 0, self.RadiusPx*self.HubRadiusPct/100.)
        PtsYBotPx = np.insert(self.RadiusPx*self.PtsYBotPct/100., 0, -self.RadiusPx*self.BladeStartYBotPct/100.)
        # Edge of the blade
        # If it coincides, then it's not a bullnose prop
        if(np.abs(PtsXTopPx[-1]-PtsXBotPx[-1]) + np.abs(PtsYTopPx[-1]-PtsYBotPx[-1]) <= 1e-3):
            PtsXBotPx[-1] = PtsXTopPx[-1]
            PtsYBotPx[-1] = PtsYTopPx[-1]
            outTop = None
        else:
            PtsAllEdge = [np.array([PtsXTopPx[-1], (PtsXTopPx[-1]+PtsXBotPx[-1])/2., PtsXBotPx[-1]]), \
                          np.array([PtsYTopPx[-1], (PtsYTopPx[-1]+PtsYBotPx[-1])/2., PtsYBotPx[-1]])]
            tckEdge, uEdge = interpolate.splprep(PtsAllEdge,k=2,s=0)
            uEdge = np.linspace(0,1,num=self.RadiusPx,endpoint=True)
            outEdge = interpolate.splev(uEdge, tckEdge)
            
        PtsAllTop = [PtsXTopPx, PtsYTopPx]
        tckTop, uTop = interpolate.splprep(PtsAllTop,k=3,s=0)
        uTop = np.linspace(0,1,num=self.RadiusPx,endpoint=True)
        outTop = interpolate.splev(uTop, tckTop)

        PtsAllBot = [PtsXBotPx, PtsYBotPx]
        tckBot, uBot = interpolate.splprep(PtsAllBot,k=3,s=0)
        uBot = np.linspace(0,1,num=self.RadiusPx,endpoint=True)
        outBot = interpolate.splev(uBot, tckBot)

        # Counterclock wise starting from center
        PtsXHubPx = np.array([0, 0, self.RadiusPx, 0, 0])*self.HubRadiusPct/100.
        PtsXHubPx = np.insert(PtsXHubPx, 2, PtsXTopPx[0])
        PtsXHubPx = np.insert(PtsXHubPx, 4, PtsXBotPx[0])
        PtsYHubPx = np.array([0, self.RadiusPx, 0, -self.RadiusPx, 0])*self.HubRadiusPct/100.
        PtsYHubPx = np.insert(PtsYHubPx, 2, PtsYTopPx[0])
        PtsYHubPx = np.insert(PtsYHubPx, 4, PtsYBotPx[0])
        PtsAllHub = [PtsXHubPx, PtsYHubPx]
        tckHub, uHub = interpolate.splprep(PtsAllHub,k=3,s=0)
        uHub = np.linspace(0,1,num=self.RadiusPx,endpoint=True)
        outHub = interpolate.splev(uHub, tckHub)

        # Top line
        self.Img[outTop[1].astype(int) + int(self.ImageSize[0]/2.), outTop[0].astype(int) + int(self.ImageSize[1]/2.)] = 255
        # Bottom line
        self.Img[outBot[1].astype(int) + int(self.ImageSize[0]/2.), outBot[0].astype(int) + int(self.ImageSize[1]/2.)] = 255
        # Hub
        self.Img[outHub[1].astype(int) + int(self.ImageSize[0]/2.), outHub[0].astype(int) + int(self.ImageSize[1]/2.)] = 255
        # Edge
        if(outEdge is not None):
            self.Img[outEdge[1].astype(int) + int(self.ImageSize[0]/2.), outEdge[0].astype(int) + int(self.ImageSize[1]/2.)] = 255
        # Dilate to not have holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.Img = cv2.dilate(self.Img, kernel)
        # Flood fill the hub
        retval, image, mask, rect = cv2.floodFill(self.Img.astype(np.uint8), None, (int(self.ImageSize[0]/2.+7),  int(self.ImageSize[1]/2.)), 255)
        # Flood fill the propeller blade
        retval, image, mask, rect = cv2.floodFill(image, None, (int(PtsYHubPx[1])+ int(self.ImageSize[0]/2.)+7,  int(PtsXHubPx[1])+ int(self.ImageSize[1]/2.)), 255)
        self.Img = image

        # Debugging tool!
        # image = cv2.circle(self.Img, (int(PtsYHubPx[1])+ int(self.ImageSize[0]/2.)+7,  int(PtsXHubPx[1])+ int(self.ImageSize[1]/2.)), radius=10, color=(255, 255, 255), thickness=-1)

        # Flip it upside down since Y is inverted
        self.Img = np.flipud(self.Img)

        # Blades are located at 360/NumBlades degrees
        ImgOneBlade = Image.fromarray(self.Img.astype(np.uint8))
        for count in range(1, self.NumBlades):
            RotImgNow = ImgOneBlade.rotate(count*360/(self.NumBlades), center=(int(self.ImageSize[0]/2.), int(self.ImageSize[0]/2.)))
            # cv2.imshow(str(count), self.Img)
            # cv2.waitKey(0)
            self.Img = cv2.bitwise_or(self.Img.astype(np.uint8), np.array(RotImgNow))
            
        # image = self.Img
        # cv2.imshow('Img', self.Img)
        # cv2.waitKey(1)

        # plt.figure()
        # if(outEdge is not None):
        #     plt.plot(PtsXTopPx, PtsYTopPx, 'ro', outTop[0], outTop[1], 'b', PtsXBotPx, PtsYBotPx,\
        #              'ko', outBot[0], outBot[1], 'g', PtsXHubPx, PtsYHubPx, 'yo', outHub[0], outHub[1], 'm', PtsAllEdge[0], PtsAllEdge[1], 'co', outEdge[0], outEdge[1], 'c')
        # else:
        #     plt.plot(PtsXTopPx, PtsYTopPx, 'ro', outTop[0], outTop[1], 'b', PtsXBotPx, PtsYBotPx,\
        #              'ko', outBot[0], outBot[1], 'g', PtsXHubPx, PtsYHubPx, 'yo', outHub[0], outHub[1], 'm')
        # plt.legend(['Points', 'Interpolated B-spline', 'True'],loc='best')
        # # plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
        # plt.axis('equal')
        # plt.title('B-Spline interpolation')
        # plt.show()

    def GenIdealFrame(self, RPM, dT):
         if(not np.any(self.Img)):
             self.GenOneProp()
         DegPerSec = RPM/60.*360.
         # Randomize clockwise and anticlockwise rotation
         Dir = np.sign(np.random.uniform() - 0.5)
         DegBetweenFrames = DegPerSec*dT
         RandAng = random.uniform(0, 360)
         I1 = Image.fromarray(self.Img)
         if(Dir < 0):
             I1 = Image.fromarray(np.fliplr(self.Img))
             DegBetweenFrames *= -1.
         # Rotate about center of the propeller
         I1 = I1.rotate(RandAng, center=(int(self.ImageSize[0]/2.), int(self.ImageSize[0]/2.)))
         I2 = I1.rotate(DegBetweenFrames, center=(int(self.ImageSize[0]/2.), int(self.ImageSize[0]/2.)))
         
         return (np.array(I1), np.array(I2))
     

    def AddBackground(self, I, PropColor, BgImgPath):
        # PropColor lies in [0, 255]
        # Deals with grayscale only
        BgImg = cv2.imread(BgImgPath, 0)

        # Resize image to required size
        try:
            BgImg = cv2.resize(BgImg, tuple(self.ImageSize))
        except:
            return None
        if(BgImg is None):
            return None
        BgImg[I == 255] = PropColor
        return BgImg
        
    @staticmethod
    def SimpleDVSGen(I1, I2, Tau):
        # Tau is percentage away from mean to threshold the values
        I1[I1==0] = 1. # So that log becomes zero
        I2[I2==0] = 1. # So that log becomes zero
        
        Vals = np.ndarray.flatten(np.log(I1)-np.log(I2))
        Vals = Vals[~np.isnan(Vals)]
        Vals = Vals[~np.isinf(Vals)]
        Vals = Vals[Vals!=0]
        Vals = Vals + 1e-3
        try:
            # plt.hist(Vals, bins=100)
            # plt.plot(Vals)
            # plt.show()
            pass
        except:
            pass

        # print(np.max(np.abs(np.log(I1+1.)-np.log(I2+1.))))

        # Not empty: hence compute gaussian
        if(Vals.any()):
            Mean = np.nanmean(Vals)
            Std = np.nanstd(Vals)
            # print(Mean, Std)
                            
            # Check if gaussian was a good fit
            if(Std >= 100.):
                EventImg = np.sign(np.log(I1)-np.log(I2))*(np.abs(np.log(I1)-np.log(I2))>=0.)
            else:
                EventImg = np.sign(np.log(I1)-np.log(I2))*(np.abs(np.log(I1)-np.log(I2))>=Tau/100.*Std)
            
            EventImg[np.isnan(EventImg)] = 0
            EventImg[np.isinf(EventImg)] = 0
            EventImg[EventImg == -1] = 127
            EventImg[EventImg == 1] = 255

            # Like Matlab's imagesc
            # plt.matshow(np.float32(np.abs(np.log(I1)-np.log(I2))))
            # plt.show()

        else:
            EventImg = np.zeros(np.shape(I1))
            
        return EventImg.astype(np.uint8)

    def GenEventFrameWithMultipleRandProps(self, MaxNumProps, MinNumProps, MaxNumBlades, MaxPropOverlapRatio,\
                                           MaxPropSizePx, MinPropSizePx, NoiseProb, MinRPM, MaxRPM, dT, Tau, BgImgPath,\
                                           MinPropColor, MaxPropColor, MaxRhoPct, CenterRadPx, Vis=False):

        # TODO: Add affine transforms to props
        RandNumProps = np.random.randint(MinNumProps, MaxNumProps+1)
        RandNumBlades = np.random.randint(2, MaxNumBlades+1, RandNumProps)

        RandPropSizes = np.random.randint(MinPropSizePx, MaxPropSizePx+1, RandNumProps)
        RandRPM = np.random.randint(MinRPM, MaxRPM+1, RandNumProps)
        RandPropColor = np.random.randint(MinPropColor, MaxPropColor+1, RandNumProps)

        ImgNow = np.zeros(tuple(self.ImageSize))
        MaskAll = np.zeros(tuple(self.ImageSize))
        MaskCentersAll = np.zeros(tuple(self.ImageSize))
        BgAll = []
        count  = 0
        while count < RandNumProps:
            # Generate each prop
            # Write parameters to self variables to be used by GenOneProp
            self.NumBlades = RandNumBlades[count]
            # TODO: Other prop variables 
            self.GenOneProp()
            I1Now, I2Now = self.GenIdealFrame(RandRPM[count], dT)
            IDisp = np.stack((I1Now, np.zeros(np.shape(I1Now)), I2Now), axis=2)

            # cv2.imshow('I1 and I2 overlayed %d'%(count), IDisp)
            # cv2.waitKey(1)

            # If you just pass a string skip this
            # You want to use the same Bg for both I1 and I2
            if(~isinstance(BgImgPath, list)):
                if(len(BgImgPath)>1):
                    # Pick a random background image
                    BgImgPathNow = BgImgPath[np.random.randint(0, len(BgImgPath))]
                else:
                    BgImgPathNow = BgImgPath[0]

            I1BgNow = self.AddBackground(I1Now, RandPropColor[count], BgImgPathNow)
            I2BgNow = self.AddBackground(I2Now, RandPropColor[count], BgImgPathNow)

            if(I1BgNow is not None):
                # cv2.imshow('I1Bg, I2Bg %d'%(count), np.hstack((I1BgNow, I2BgNow)))
                # cv2.waitKey(1)

                EventImgBgNow = self.SimpleDVSGen(I1BgNow, I2BgNow, Tau)

                # Assumes square image of propeller
                RandCenterX = np.random.randint(RandPropSizes[count], self.ImageSize[0]-RandPropSizes[count]-1)
                RandCenterY = np.random.randint(RandPropSizes[count], self.ImageSize[1]-RandPropSizes[count]-1)

                MaskNow = np.zeros(tuple(self.ImageSize))
                CenterX = self.ImageSize[0]/2.
                CenterY = self.ImageSize[1]/2.
                MaskNow[int(np.ceil(CenterX-RandPropSizes[count]/2.)):int(np.ceil(CenterX+RandPropSizes[count]/2.)),\
                  int(np.ceil(CenterY-RandPropSizes[count]/2.)):int(np.ceil(CenterY+RandPropSizes[count]/2.))] = 255

                
                # Warp each prop with homography
                WarpedI, WarpedMask, WarpedMaskCenter = self.RandHomographyWarp(EventImgBgNow, MaxRhoPct, CenterRadPx, Vis=False)

                # Dont overwrite previous data
                MaskImgNowCrop = ImgNow[int(np.ceil(RandCenterX-RandPropSizes[count]/2.)):int(np.ceil(RandCenterX+RandPropSizes[count]/2.)),\
                  int(np.ceil(RandCenterY-RandPropSizes[count]/2.)):int(np.ceil(RandCenterY+RandPropSizes[count]/2.))] == 0
                
                ImgNow[int(np.ceil(RandCenterX-RandPropSizes[count]/2.)):int(np.ceil(RandCenterX+RandPropSizes[count]/2.)),\
                  int(np.ceil(RandCenterY-RandPropSizes[count]/2.)):int(np.ceil(RandCenterY+RandPropSizes[count]/2.))] +=\
                      (cv2.resize(WarpedI, (RandPropSizes[count], RandPropSizes[count])))*MaskImgNowCrop


                MaskCrop = MaskAll[int(np.ceil(RandCenterX-RandPropSizes[count]/2.)):int(np.ceil(RandCenterX+RandPropSizes[count]/2.)),\
                  int(np.ceil(RandCenterY-RandPropSizes[count]/2.)):int(np.ceil(RandCenterY+RandPropSizes[count]/2.))]

                MaskCrop = ((MaskCrop>0) | (cv2.resize(WarpedMask, (RandPropSizes[count], RandPropSizes[count]))>0))*255.
                
                MaskAll[int(np.ceil(RandCenterX-RandPropSizes[count]/2.)):int(np.ceil(RandCenterX+RandPropSizes[count]/2.)),\
                  int(np.ceil(RandCenterY-RandPropSizes[count]/2.)):int(np.ceil(RandCenterY+RandPropSizes[count]/2.))] = MaskCrop

                MaskCentersAllMask = (MaskCentersAll[int(np.ceil(RandCenterX-RandPropSizes[count]/2.)):int(np.ceil(RandCenterX+RandPropSizes[count]/2.)),\
                  int(np.ceil(RandCenterY-RandPropSizes[count]/2.)):int(np.ceil(RandCenterY+RandPropSizes[count]/2.))] == 0)
                MaskCentersAll[int(np.ceil(RandCenterX-RandPropSizes[count]/2.)):int(np.ceil(RandCenterX+RandPropSizes[count]/2.)),\
                  int(np.ceil(RandCenterY-RandPropSizes[count]/2.)):int(np.ceil(RandCenterY+RandPropSizes[count]/2.))] +=\
                       cv2.resize(WarpedMaskCenter, (RandPropSizes[count], RandPropSizes[count]))*MaskCentersAllMask
                

                if(Vis):
                    # plt.matshow(np.float32(ImgNow))
                    # plt.show()
                    cv2.imshow('ImgNow MaskNow %d'%(count), np.concatenate((ImgNow.astype(np.uint8), MaskAll.astype(np.uint8), MaskCentersAll.astype(np.uint8)), axis=1))               
                    cv2.waitKey(0)

                # Detect corners to generate Gaussian label
                CornerLabel = cv2.dilate(MaskCentersAll.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
                Sigma = (self.HubRadiusPct*(self.RadiusPx/MaxPropSizePx)/2.)
                CornerLabelGaussian = cv2.GaussianBlur(CornerLabel, (5,5), Sigma)

                BgAll.append(BgImgPathNow)
                count += 1
                

                if(Vis):
                    cv2.imshow('Corners', CornerLabel)
                    cv2.imshow('CornersBlur', CornerLabelGaussian.astype(np.uint8))
                    cv2.waitKey(0)

            NoiseMask = np.zeros_like(ImgNow)
            NoiseMask = (np.random.uniform(size=(np.shape(NoiseMask)[0], np.shape(NoiseMask)[1])) < NoiseProb)*255.
            NoiseMaskSign = (np.random.uniform(size=(np.shape(NoiseMask)[0], np.shape(NoiseMask)[1])) >= 0.5) # False means -ve event and True means +ve event
            NoiseMask[~NoiseMaskSign*(NoiseMask==255)] = 127.
            NoiseMask[NoiseMaskSign*(NoiseMask==255)] = 255.
            NoiseMask = NoiseMask.astype(np.uint8)

        ImgNow[NoiseMask > 0] = NoiseMask[NoiseMask > 0]
        # Convert to +1 and -1
        ImgMask = ImgNow>0
        ImgNowDiff1 = np.abs(ImgMask*(ImgNow-127.))
        ImgNowDiff2 = np.abs(ImgMask*(ImgNow-255.))
        DiffThld = 50
        # Remove small values from interpolation artifacts
        ImgNowDiff1[ImgNowDiff1 < DiffThld] = 0.
        ImgNowDiff2[ImgNowDiff2 < DiffThld] = 0.
        ImgPol = np.zeros_like(ImgNow)
        ImgPol[ImgNowDiff1 > ImgNowDiff2] = 1
        ImgPol[ImgNowDiff1 <= ImgNowDiff2] = -1
        ImgPol = ImgPol*ImgMask
        
        return ImgNow.astype(np.uint8), ImgPol, MaskAll.astype(np.uint8), CornerLabelGaussian.astype(np.uint8), BgAll
        
        

    def RandHomographyWarp(self, I, MaxRhoPct, CenterRadPx, Vis=False):
        # MaxRhoPct is max perturbation in Pct of Prop Size (Cannot be more than 50)
        # Pad to ImageSize so we don't miss any data even in extreme warps
        OrgImageSize = np.shape(I)
        Mask = (np.ones_like(I)*255.).astype(np.uint8)
        I = np.pad(I, 2*np.max((OrgImageSize[0], OrgImageSize[1])), mode='constant', constant_values=(0, 0))
        Mask = np.pad(Mask, 2*np.max((OrgImageSize[0], OrgImageSize[1])), mode='constant', constant_values=(0, 0))
        # Center label
        MaskCenter = np.zeros_like(Mask).astype(np.uint8)
        MaskCenter = cv2.circle(MaskCenter, (int(np.shape(MaskCenter)[0]/2.), int(np.shape(MaskCenter)[0]/2.)),\
                                    CenterRadPx, color=(255,255,255), thickness=-1) # thicknesss -1 gives filled circle

        ImageSize = np.shape(I)

        CenterX = int(ImageSize[0]/2.-OrgImageSize[0]/2.)
        CenterY = int(ImageSize[1]/2.-OrgImageSize[1]/2.)
        
        p1 = (CenterX, CenterY)
        p2 = (CenterX, CenterY + OrgImageSize[1])
        p3 = (CenterX + OrgImageSize[0], CenterY + OrgImageSize[1])
        p4 = (CenterX + OrgImageSize[0], CenterY)

        AllPts = [p1, p2, p3, p4]

        PerturbPts = []
        Rho = MaxRhoPct/100.*np.min(OrgImageSize)
        for point in AllPts:
           PerturbPts.append((point[0] + np.random.uniform(-Rho,Rho), point[1] + np.random.uniform(-Rho,Rho)))

        # Obtain Homography between the 2 images
        H = cv2.getPerspectiveTransform(np.float32(AllPts), np.float32(PerturbPts))
        # Get Inverse Homography
        HInv = np.linalg.inv(H)

        WarpedI = cv2.warpPerspective(I, HInv, (ImageSize[1],ImageSize[0]))
        WarpedMask = cv2.warpPerspective(Mask, HInv, (ImageSize[1],ImageSize[0]))
        WarpedMaskCenter = cv2.warpPerspective(MaskCenter, HInv, (ImageSize[1],ImageSize[0]))

        if(Vis):
            IDisp =  np.stack((I, np.zeros(np.shape(I)), WarpedI), axis=2)
            cv2.imshow('Overlayed', cv2.resize(IDisp.astype(np.uint8), tuple(OrgImageSize)))
            cv2.waitKey(0)

        # Crop to valid values
        ImageSize = np.shape(I)
        CenterX = ImageSize[0]/2.
        CenterY = ImageSize[1]/2.
        OutShape = OrgImageSize
        # https://stackoverflow.com/questions/60692394/python-how-to-crop-an-image-according-to-its-values
        x, y = np.nonzero(WarpedMask)
        xl,xr = x.min(),x.max()
        yl,yr = y.min(),y.max()
        WarpedI = WarpedI[xl:xr+1, yl:yr+1]
        WarpedMask = WarpedMask[xl:xr+1, yl:yr+1]
        WarpedMaskCenter = WarpedMaskCenter[xl:xr+1, yl:yr+1]
        
        # Resize back to original size
        WarpedI = cv2.resize(WarpedI.astype(np.uint8), tuple(OrgImageSize))
        WarpedMask = cv2.resize(WarpedMask.astype(np.uint8), tuple(OrgImageSize))
        WarpedMaskCenter = cv2.resize(WarpedMaskCenter.astype(np.uint8), tuple(OrgImageSize))

        if(Vis):
            cv2.imshow('WarpedICrop', WarpedI.astype(np.uint8))
            cv2.imshow('WarpedMaskCrop', WarpedMask.astype(np.uint8))
            cv2.imshow('WarpedMaskCenterCrop', WarpedMaskCenter.astype(np.uint8))
            cv2.waitKey(0)

            
        return WarpedI, WarpedMask, WarpedMaskCenter

def GenerateData(Args):

    pg = PropGen(NumBlades = 4, ImageSize = [480, 480], RadiusPx = 200)

    MaxNumProps = 12
    MinNumProps = 1
    MaxNumBlades = 5
    MaxPropOverlapRatio = 0.5 # Not Used!
    MinPropSizePx = 40
    MaxPropSizePx = 120
    NoiseProb = 0.02
    MinRPM = 5000.
    MaxRPM = 30000.
    dT = 1e-4 # in secs
    Tau = 80
    BgImgPath =  glob.glob(Args.BgImgPath + '*.' + Args.ImgFormat)
    MinPropColor = 0
    MaxPropColor = 255
    MaxRhoPct = 30.
    CenterRadPx = 15

    BgNames = open(Args.WritePath+ 'BgNames.txt', 'w')
    for count in tqdm(range(Args.NumImages)):
        Img, ImgPol, Mask, CornerLabelGaussian, Bg = pg.GenEventFrameWithMultipleRandProps(MaxNumProps, MinNumProps, MaxNumBlades,\
                                                                                           MaxPropOverlapRatio, MaxPropSizePx,\
                                                                                           MinPropSizePx, NoiseProb, MinRPM, MaxRPM,\
                                                                                           dT, Tau, BgImgPath, MinPropColor, MaxPropColor,\
                                                                                           MaxRhoPct, CenterRadPx)
        
        
        FileNameNow = '%06d'%(count) + '.png'
        ImgPol = iu.PolImgDisp(ImgPol)
        Overlayed = iu.OverlayCornersOnImg(Img, CornerLabelGaussian)

        cv2.imwrite(Args.WritePath+'Imgs/'+FileNameNow, ImgPol)
        cv2.imwrite(Args.WritePath+'Masks/'+FileNameNow, Mask)
        cv2.imwrite(Args.WritePath+'Labels/'+FileNameNow, CornerLabelGaussian)
        cv2.imwrite(Args.WritePath+'Overlays/'+FileNameNow, Overlayed)
        BgNames.write(','.join(Bg)+'\n')
        

        # cv2.imshow('Generated Img, Mask, Label', np.concatenate((Img, ImgPol, Mask, CornerLabelGaussian, Overlayed), axis=1))
        # cv2.waitKey(0)

if __name__=="__main__":
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BgImgPath', default='/home/nitin/Datasets/MSCOCO/train2014/', help='Path from where background images are read, Default:/home/nitin/Datasets/MSCOCO/train2014/')
    Parser.add_argument('--ImgFormat', default='jpg', help='Image format, Default: jpg')
    Parser.add_argument('--NumImages', type=int, default=10, help='Number of images to generate, Default:10')
    Parser.add_argument('--WritePath', default='/home/nitin/Datasets/DVSProp/', help='Path to save images, Default:/home/nitin/Datasets/DVSProp')
    # Parser.add_argument('--MaxNumProps', type=int, default=12, help='Maximum number of propellers in an image, Default:12')
    # Parser.add_argument('--MinNumProps', type=int, default=1, help='Minimum number of propellers in an image, Default:1')
    # Parser.add_argument('--MaxNumBlades', type=int, default=5, help='Maximum number of blades for each propeller (minimum is 2), Default:5')
    
    Args = Parser.parse_args()

    # If WritePath doesn't exist make the path
    if(not (os.path.isdir(Args.WritePath))):
       os.makedirs(Args.WritePath)
       os.makedirs(Args.WritePath+'Imgs')
       os.makedirs(Args.WritePath+'Masks')
       os.makedirs(Args.WritePath+'Labels')
       os.makedirs(Args.WritePath+'Overlays')
       
    GenerateData(Args)

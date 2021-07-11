clc
clear all
close all

ReadPath = 'C:/Users/Nitin/Desktop/Frames/';

Dirs = dir([ReadPath, '*.png']);
Tau = 2;
% 2 for Frames, 151
% 0.7 for Frames2, 41

StartIdx = 160;

for count = StartIdx:length(Dirs)-1
    FileName = [ReadPath, 'frame', sprintf('%06d',count), '.png'];
    if(count == StartIdx)
        I1 = double(rgb2gray(imread(FileName)));
    else
        I2 = double(rgb2gray(imread(FileName)));
        FrameDiff = log(I2) - log(I1);
        FrameDiff(isinf(FrameDiff)) = 0;
        PosEvents = FrameDiff >= Tau;
        NegEvents = FrameDiff <= -Tau;
        EventImgR = zeros(size(I1));
        EventImgG = zeros(size(I1));
        EventImgB = zeros(size(I1));
        EventImgR(PosEvents) = 255;
        EventImgB(NegEvents) = 255;
        EventImg = cat(3, EventImgR, EventImgG, EventImgB);
        %         figure,
        %         imagesc(FrameDiff);
        %         colormap jet
        %         colorbar
        %         figure,
        imshow(EventImg);
        I1 = I2;
    end
    title(count);
    drawnow;
    % pause(0.1);
end

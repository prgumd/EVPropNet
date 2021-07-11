clc
clear all
close all

VidPath = 'C:/Users/Nitin/Desktop/SnowFalling.mp4';
WritePath = 'C:/Users/Nitin/Desktop/Frames2/';
if(~exist(WritePath, 'dir'))
    mkdir(WritePath);
end
v = VideoReader(VidPath);

count = 0;
% targetSize = [128, 128, 3];

while hasFrame(v)
    frame = readFrame(v);
    % frame = imresize(frame, 0.5);
%     frame = frame(ceil(size(frame,1)/2-targetSize(1)/2):floor(size(frame,1)/2+targetSize(1)/2-1),...
%                   ceil(size(frame,2)/2-targetSize(2)/2):floor(size(frame,2)/2+targetSize(2)/2-1), :);
    FileName = [WritePath, 'frame', sprintf('%06d',count), '.png'];
    imwrite(frame, FileName);
    disp(count);
    count = count + 1; 
end
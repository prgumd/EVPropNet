clc
clear all
close all

BasePath = '/media/nitin/Research/QuadDVS/ForPaper/LandingAndFollow/';
FileName = 'followtest2Crop.csv';
WriteBasePath = '/media/nitin/Research/QuadDVS/ForPaper/LandingAndFollow/Images2/';
WriteFileName = strsplit(FileName, '.csv');
WriteFileName = WriteFileName{1};
WritePath = [WriteBasePath, WriteFileName];
Vis = 1;


Data = table2array(readtable([BasePath, FileName]));
disp('Data Loading Complete ....');

dT = 0.0004; % Integration time in secs
StartIdx = 1;

ImageSize = [480, 640, 3];

WriteFlag = 1;
WriteFrames = 1e5;
WriteImageSize = [480, 480];


if(~exist(WritePath, 'dir'))
    mkdir(WritePath)
end


TPrev = 0;
FrameCount = 1;
Frames = {};
FrameNow = zeros(ImageSize);
NumSamples = zeros(ImageSize(1), ImageSize(2));

WriteCount = 0;

for count = StartIdx:size(Data,1)
    % Data is (x, y, pol, t)
    if(Data(count, 4) - TPrev >= dT)
        FrameNow = FrameNow./repmat(NumSamples, 1, 1, 3);
        if(WriteFlag && (WriteCount < WriteFrames))
            PositiveEvents = FrameNow(:,:,1)>0;
            NegativeEvents = FrameNow(:,:,2)>0;
            FrameWriteNow = ones(ImageSize(1), ImageSize(2))*127;
            FrameWriteNow(PositiveEvents) = 255;
            FrameWriteNow(NegativeEvents) = 0;
            FrameWriteNow = uint8(FrameWriteNow);
            FrameWriteNow = repmat(FrameWriteNow, 1,1,3);
            if(Vis)
                imshow(FrameWriteNow);
                pause(0.001);
                drawnow;
            end
            imwrite(FrameWriteNow, [WritePath, '/', sprintf('%06d', WriteCount), '.png']);
            disp([WritePath, '/', sprintf('%06d', WriteCount), '.png']);
            WriteCount = WriteCount + 1;
        else
            pause;
            if(Vis)
                
                imshow(FrameNow);
                title(FrameCount);
                pause(0.001);
                drawnow;
            end
        end
        FrameCount = FrameCount + 1;
        TPrev = Data(count, 4);
        FrameNow = zeros(ImageSize);
        NumSamples = zeros(ImageSize(1), ImageSize(2));
    end
    FrameNow(Data(count,2)+1, Data(count,1)+1, Data(count,3)+1) = 1;
    NumSamples(Data(count,2)+1, Data(count,1)+1) = NumSamples(Data(count,2)+1, Data(count,1)+1) + 1;
    disp(count./size(Data,1)*100);
end

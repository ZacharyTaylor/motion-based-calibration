function [ projectedLidar, validPoints ] = projectLidar( tform, cam, lidar, imageSize )
%PROJECTLIDAR Summary of this function goes here
%   Detailed explanation goes here

%% check inputs
if(~isequal(size(tform),[4,4]))
    error('tform must be 4 by 4');
end
if(~existsOnGPU(tform))
    error('tform must be on GPU');
end
if(~strcmp(classUnderlying(tform),'single'))
    error('tform must be of type single');
end

if(~isequal(size(cam),[3,4]))
    error('cam must be 3 by 4');
end
if(~existsOnGPU(cam))
    error('cam must be on GPU');
end
if(~strcmp(classUnderlying(cam),'single'))
    error('cam must be of type single');
end

if(size(lidar,2)< 3)
    error('lidar must be atleast x by 3');
end
if(~existsOnGPU(lidar))
    error('lidar must be on GPU');
end
if(~strcmp(classUnderlying(lidar),'single'))
    error('lidar must be of type single');
end

if(~isequal(size(imageSize),[1,2]))
    error('imageSize must be 1 by 2');
end
imageSize = uint32(imageSize);

[validPoints, projectedLidar] = projectLidarMex(tform, cam, lidar(:,1:3), imageSize);
projectedLidar = [projectedLidar, lidar(:,4:end)];
projectedLidar = projectedLidar(validPoints,:);

end


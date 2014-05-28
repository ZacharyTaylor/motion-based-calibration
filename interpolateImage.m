function [ intVals ] = interpolateImage(image, points)
%PROJECTLIDAR Summary of this function goes here
%   Detailed explanation goes here

%% check inputs
if(~existsOnGPU(image))
    error('image must be on GPU');
end
if(~strcmp(classUnderlying(image),'single'))
    error('image must be of type single');
end

if(size(points,2)< 2)
    error('points must be atleast x by 2');
end
if(~existsOnGPU(points))
    error('lidar must be on GPU');
end
if(~strcmp(classUnderlying(points),'single'))
    error('points must be of type single');
end

intVals = interpolateImageMex(image, points);

end


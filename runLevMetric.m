function [ error ] = runLevMetric( tform, K, lidar, images )
%RUNFLOWMETRIC Summary of this function goes here
%   Detailed explanation goes here

T = zeros(4,4);
T(1:3,1:3) = angle2dcm(tform(1),tform(2),tform(3));
T(4,4) = 1;
T(1:3,4) = tform(4:6)';
T = gpuArray(single(T));

ims = size(images{1});

error = cell(size(lidar,1),1);
for i = 1:size(lidar,1)
    points = projectLidar(T, K, lidar{i}, ims(1:2));
    intVals = interpolateImage(images{i}, points(:,1:2));
    error{i} = evalLev(intVals,points(:,3));
end

error = cell2mat(error);
error = -sum(error,1);

a = 100*rand(1);
if(a > 99)
    t = 180*tform(1:3)/pi;
    fprintf('R: %f P: %f, Y: %f, X: %f, Y: %f, Z: %f, Err: %f\n',t(1),t(2),t(3),tform(4),tform(5),tform(6),error);
end


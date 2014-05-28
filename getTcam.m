function [ Tcam, C1, C1BaseValid, C2] = getTcam( im1, im2, mask, K, C1Base )
%GETTCAM Gets normalized camera transform given two images and K
% inputs:
%   im1- 1st image
%   im2- image to match points in im1 to
%   mask- mask size of im1 and 2 to remove any parts of vechile in view of
%   the sensor
%   K- camera matrix
%   C1Base- keypoints already located in image 1 that should be tracked
%
% outputs:
%   Tcam- normalized camera transform
%   points- 3d point locations as seen from position of im1
%   C1- location of points in 1st image
%   C1BaseValid- points in C1Base that are still valid (these points make
%   up the start of C1 and C2)
%   C2- location of points in 2nd image

    tracker = vision.PointTracker('MaxBidirectionalError', 1);
    
    %detect points (lots of feature options SURF appears to give best
    %balance between speed and robustness)
    %C = detectFASTFeatures(im1);
    %C = detectMinEigenFeatures(im1);
    %C = detectHarrisFeatures(im1);
    C1 = detectSURFFeatures(im1);
    
    C1 = double(C1.Location);
    
    C1BaseValid = [(1:size(C1Base,1))';zeros(size(C1,1),1)];
    
    %add base points
    C1 = [C1Base;C1];
    
    %remove masked points
    notMasked = mask(round(C1(:,2))+size(mask,1)*(round(C1(:,1))-1)) ~= 0;
    C1 = C1(notMasked,:);
    C1BaseValid = C1BaseValid(notMasked,:);

    %match points
    initialize(tracker, C1, im1);
    [C2, tracked] = step(tracker, im2);
    release(tracker);
    
    %remove untracked points
    C1 = C1(tracked,:);
    C2 = C2(tracked,:);
    C1BaseValid = C1BaseValid(tracked,:);
    
    %get fundemental matrix
    [F, inliers] = estimateFundamentalMatrix(C1,C2,'NumTrials',500);
    C1 = C1(inliers,:);
    C2 = C2(inliers,:);
    C1BaseValid = C1BaseValid(inliers,:);
    
    %get essential matrix
    E = K'*F*K;
    
    %Hartley matrices
    W = [0 -1 0; 1 0 0; 0 0 1];
       
    %get P
    [U,~,V] = svd(E);
    RA = U*W*V';
    RB = U*W'*V';
    
    RA = RA*sign(RA(1,1));
    RB = RB*sign(RB(1,1));
    
    TA = U(:,3);
    TB = -U(:,3);

    %get Transform
    Tcam = zeros(4,4,2);

    if sum(diag(RA)) > sum(diag(RB))
        Tcam(:,:,1) = [[RA,TA];[0,0,0,1]];
        Tcam(:,:,2) = [[RA,TB];[0,0,0,1]];
    else
        Tcam(:,:,1) = [[RB,TA];[0,0,0,1]];
        Tcam(:,:,2) = [[RB,TB];[0,0,0,1]];
    end
        
    best = zeros(2,1);
    
    %match points
    points = zeros(size(C1,1),3,2);
    for j = 1:2
        P1 = [K,[0;0;0]];
        P2 = [K,[0;0;0]]*Tcam(:,:,j);
        for i = 1:size(C1,1)

            A = zeros(4,4);
            A(1,:) = C1(i,1)*P1(3,:)' - P1(1,:)';
            A(2,:) = C1(i,2)*P1(3,:)' - P1(2,:)';
            A(3,:) = C2(i,1)*P2(3,:)' - P2(1,:)';
            A(4,:) = C2(i,2)*P2(3,:)' - P2(2,:)';

            [~, ~, V] = svd(A);
            points(i,:,j) = V(1:3,4)'/V(4,4);
        end

        best(j,1) = sum((sign(points(:,3,j))));
    end
    
    [~,idx] = max(best);
    points = points(:,:,j);
    Tcam = Tcam(:,:,idx);
    
    %filter out negitive and distant point matches
    badPoints = or(sqrt(sum(points.^2,2)) > 1000, points(:,3) < 0);
    C1 = C1(~badPoints,:);
    C2 = C2(~badPoints,:);
    C1BaseValid = C1BaseValid(~badPoints,:);
    
    %get valid base points
    C1BaseValid = C1BaseValid(C1BaseValid ~= 0);
end


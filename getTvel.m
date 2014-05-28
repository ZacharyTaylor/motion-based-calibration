function [ Tvel ] = getTvel( vel1, vel2 )
%GETTCAM Gets normalized camera transform given two images and K
% optionally also generates estimated variance in second output argument
     
%get points less then 40 meters away
dist = sqrt(sum(vel1(:,1:3).^2,2));
vel1 = vel1(dist < 40,:);

dist = sqrt(sum(vel2(:,1:3).^2,2));
vel2 = vel2(dist < 40,:);

%thin if more then 20000 points
if(size(vel1,1) > 20000)
    vel1 = datasample(vel1,20000,'Replace',false);
end
if(size(vel2,1) > 20000)
    vel2 = datasample(vel2,20000,'Replace',false);
end

Tvel = icpMex(vel2(:,1:3)',vel1(:,1:3)',eye(4),1,'point_to_point');

end


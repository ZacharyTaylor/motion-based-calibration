function [ points ] = LevLidar(cloud)
%gets distance to each point

y = 0.5;

cloud(:,4) = sqrt(cloud(:,1).^2+cloud(:,2).^2 + cloud(:,3).^2);

points = cloud(2:end-1,:);
points(:,4) = max(abs(cloud(1:end-2,4) - cloud(2:end-1,4)), abs(cloud(3:end,4) - cloud(2:end-1,4)));
points(:,4) = points(:,4).^y;

points = points(points(:,4) > 0.3,:);

end


function [ out ] = LevImage(in)
%gets distance to each point
in = double(in);

E = zeros(size(in,1)-2, size(in,2)-2);
D = zeros(size(in,1)-2, size(in,2)-2);

for x = 1:3
    for y = 1:3
        E = max(E, abs(in(y:size(in,1)+y-3,x:size(in,2)+x-3) - in(2:size(in,1)-1, 2:size(in,2)-1)));
    end
end

s = 100;

x = 0;
for y = -s:s
    dxL = max(0,x);
    dxU = min(size(E,2)-1, size(E,2)-1+x);

    dyL = max(0,y);
    dyU = min(size(E,1)-1, size(E,1)-1+y);

    temp = E(dyL+1:dyU+1, dxL+1:dxU+1).*(0.98.^(max(abs(x),abs(y)))); 

    D((dyL-y+1):(dyU-y+1), (dxL-x+1):(dxU-x+1)) = max(D((dyL-y+1):(dyU-y+1), (dxL-x+1):(dxU-x+1)), temp);

end
    
for x = -s:s
    y = 0;
    dxL = max(0,x);
    dxU = min(size(E,2)-1, size(E,2)-1+x);

    dyL = max(0,y);
    dyU = min(size(E,1)-1, size(E,1)-1+y);

    temp = E(dyL+1:dyU+1, dxL+1:dxU+1).*(0.98.^(max(abs(x),abs(y)))); 

    D((dyL-y+1):(dyU-y+1), (dxL-x+1):(dxU-x+1)) = max(D((dyL-y+1):(dyU-y+1), (dxL-x+1):(dxU-x+1)), temp);

end

D = (2*E + D)/3;

out = zeros(size(in));

out(2:size(in,1)-1,2:size(in,2)-1) = D;

end


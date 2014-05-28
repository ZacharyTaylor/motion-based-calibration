function [ Tmean, Tvar ] = calcT( Tvel, Tcam, R )
%GETTRANS Summary of this function goes here
%   Detailed explanation goes here

r = 10;

S = zeros(3*r,r+3);
X = zeros(3*r,1);

Tvar = zeros(500,3);
errMin = inf;

%remove transforms with a zero translation
in = [Tcam Tvel];
in = in(:,:,all([in(1:3,4,:);in(1:3,8,:)],1));

for i = 1:500
    data = datasample(in,r,3);
    
    for frame = 1:r
        S(3*(frame-1)+1:3*(frame-1)+3,1:3) = (data(1:3,1:3,frame)-eye(3));
        S(3*(frame-1)+1:3*(frame-1)+3,frame+3) = data(1:3,4,frame);
    
        X(3*(frame-1)+1:3*(frame-1)+3) = R*data(1:3,8,frame);
    end

    temp = S\X;
    Tvar(i,:) = temp(1:3);
    
    %find scales and error
    err = zeros(size(Tcam,3),1);
    for j = 1:size(Tcam,3)
        if(any(Tcam(1:3,4,j)))
            s = Tcam(1:3,4,j)\(R*Tvel(1:3,4,j) + Tvar(i,:)' - Tcam(1:3,1:3,j)*Tvar(i,:)');
            err(j) = sum((R*Tvel(1:3,4,j) + Tvar(i,:)' - Tcam(1:3,1:3,j)*Tvar(i,:)' - s*Tcam(1:3,4,j)).^2);
        else
            err(j) = 0;
        end
        
    end
    
    err = median(err);
    if(err < errMin)
        errMin = err;
        Tmean = Tvar(i,:);
    end
        
end

Tvar = (1.4826*mad(Tvar,1)).^2;

end


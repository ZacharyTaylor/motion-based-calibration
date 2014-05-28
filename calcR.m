function [ M, V, err ] = calcR( Tvel, Tcam)
%LMEDSR get least median of squares rotation

sample = 500;
sub = 10;

T1R = zeros(size(Tcam,3),4);
T1D = zeros(size(Tcam,3),4);
T2R = zeros(size(Tvel,3),4);
T2D = zeros(size(Tvel,3),4);

for i = 1:size(Tvel,3)
    T1R(i,:) = vrrotmat2vec(Tcam(1:3,1:3,i));
    T2R(i,:) = vrrotmat2vec(Tvel(1:3,1:3,i));

    T1R(i,1:3) = T1R(i,1:3)*T1R(i,4);
    T2R(i,1:3) = T2R(i,1:3)*T2R(i,4);
    
    T1D(i,1:3) = Tcam(1:3,4,i)./norm(Tcam(1:3,4,i));
    T2D(i,1:3) = Tvel(1:3,4,i)./norm(Tvel(1:3,4,i));
    
    T1D(i,1:3) = T1D(i,1:3);
    T2D(i,1:3) = T2D(i,1:3);
end

T1D(~isfinite(T1D(:))) = 0;
T2D(~isfinite(T2D(:))) = 0;
T1R(~isfinite(T1R(:))) = 0;
T2R(~isfinite(T2R(:))) = 0;

dV = zeros(3,3,sample);
errMinD = inf;
for i = 1:sample
    data = datasample([T2D(:,1:3) T1D(:,1:3)],sub,1);
    dV(:,:,i) = Kabsch(data(:,1:3)',data(:,4:6)');
    err = median(sum((dV(:,:,i)*T2D(:,1:3)'-T1D(:,1:3)').^2,1));
    
    if(err < errMinD)
        errMinD = err;
        dM = dV(:,:,i);
    end
    
end

dV = (1.4826*mad(dV,1,3)).^2;

rV = zeros(3,3,sample);
errMinR = inf;
for i = 1:sample
    data = datasample([T2R(:,1:3) T1R(:,1:3)],sub,1);
    rV(:,:,i) = Kabsch(data(:,1:3)',data(:,4:6)');
    err = median(sum((rV(:,:,i)*T2R(:,1:3)'-T1R(:,1:3)').^2,1));
    
    if(err < errMinR)
        errMinR = err;
        rM = rV(:,:,i);
    end
    
end

err = errMinR + errMinD;

rV = (1.4826*mad(rV,1,3)).^2;

%combine results into one measure
dW = (1./dV)./((1./rV)+(1./dV));
rW = (1./rV)./((1./rV)+(1./dV));

M = dM.*dW + rM.*rW;
V = dV.*dW + rV.*rW;

%reorthogonalize
[a,b,c] = svd(M);
temp = a*c';
if(det(temp) < 0)
    [~,idx] = min(sum(M.^2));
    M = temp;
    M(:,idx) = -M(:,idx);
end


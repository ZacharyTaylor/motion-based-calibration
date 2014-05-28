function [ solution ] = OptimizePop( T, V, K, scans, images )
%OPTIMIZE Runs particle swarm to find optimum values

numPoints = 200;

popT = repmat(T(1:3,4)',numPoints,1) + repmat(sqrt(V(1:3,4)'),numPoints,1).*randn(numPoints,3);
popT(:) = max(popT(:),repmat(-4,size(popT(:))));
popT(:) = min(popT(:),repmat(4,size(popT(:))));

tempR = repmat(T(1:3,1:3),[1,1,numPoints]) + repmat(sqrt(V(1:3,1:3)),[1,1,numPoints]).*randn(3,3,numPoints);
popR = zeros(numPoints,3);
for i = 1:numPoints
    [a,~,b] = svd(tempR(:,:,i));
    tempR(:,:,i) = a*b';
    [popR(i,1), popR(i,2), popR(i,3)] = dcm2angle(tempR(:,:,i));
end

lower = zeros(1,6);
upper = zeros(1,6);
    
pop = [popR, popT];

lower(1:3) = -5;
upper(1:3) = 5;
lower(4:6) = -pi;
upper(4:6) = pi;

options = psooptimset('PopulationSize', numPoints,...
    'TolCon', 1e-1,...
    'ConstrBoundary', 'absorb',...
    'StallGenLimit', 50,...
    'Generations', 300,...
    'InitialPopulation', pop);

warning('off','images:initSize:adjustingMag');

solution = pso(@(tform) runLevMetric( tform, K, scans, images ), 6,[],[],[],[],lower,upper,[],options);

end


function [ data] = ReadKittiVelDataSingle( path )
%READVELDATA Reads binary velodyne data

fid = fopen(path, 'r');

data = fread(fid,800000,'single');
data = reshape(data,4,[])';
fclose(fid);

end


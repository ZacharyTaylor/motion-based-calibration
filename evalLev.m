function [ error ] = evalLev( A, B )
    error = A.*B;
    error = sum(error);
    error = gather(error);
end


function out = partialTrace(X, sys, dim)
    %PARTIALTRACE Takes the partial trace of a 2-system quantum state

    arguments
        X   (:, :) double
        sys (1, 1) {mustBeInteger} = 2
        dim (1, 2) {mustBeInteger} = [sqrt(length(X)), sqrt(length(X))]
    end

    M = dim(1);
    N = dim(2);
    
    if sys == 2
        out = zeros(M);
        for i = 1:M
            for j = 1:M
                out(i, j) = trace( X((i-1)*N+1:i*N, (j-1)*N+1:j*N) );
            end
        end
    else
        out = zeros(N);
        for i = 1:M
            out = out + X((i-1)*N+1:i*N, (i-1)*N+1:i*N);
        end
    end
end


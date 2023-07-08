function out = sparsePartialTrace(X, sys, N)
    % Faster partial trace operation when we know output is diagonal
    
    arguments
        X   (:, :) double
        sys (1, 1) {mustBeInteger} = 2
        N   (1, 1) {mustBeInteger} = sqrt(length(X))
    end

    out = zeros(N, 1);
    if sys == 2
        for i = 1:N:N^2
            out((i-1)/N + 1) = sum(X(i:i+N-1, i:i+N-1), 'all');
        end
    else
        for i = 1:N
            out(i) = sum(X(i:N:N^2, i:N:N^2), 'all');
        end
    end
end
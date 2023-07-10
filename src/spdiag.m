function X = spdiag(x)
    %SPDIAG Summary of this function goes here
    %   Detailed explanation goes here
    N = length(x);
    X = sparse(1:N, 1:N, x);
end


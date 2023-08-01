function X = spdiag(x)
    %SPDIAG Creates sparse diagonal matrix with entries x
    N = length(x);
    X = sparse(1:N, 1:N, x);
end


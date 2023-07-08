function [AR, R, AR_vec] = Purify(A)
    % Purify a diagonal state A

    N = length(A);
    AR_vec = sparse(1:N+1:N^2, ones(N, 1), sqrt(A));
    AR = AR_vec * AR_vec';
    R = A;

end


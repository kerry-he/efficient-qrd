function [AR, R, AR_vec] = Purify(A)
    % Purify a diagonal state A

    N = length(A);

    if numel(A) == N
        
        AR_vec = sparse(1:N+1:N^2, ones(N, 1), sqrt(A));
        AR = AR_vec * AR_vec';
        R = A;

    else

        [U, D] = eig(A);
        AR_vec = zeros(N^2, 1);
        for i = 1:N
            r_vec = zeros(N, 1); r_vec(i) = 1;
            AR_vec = AR_vec + sqrt(D(i, i)) * kron(U(:, i), r_vec);
        end
        AR = AR_vec * AR_vec';
        R = diag(D);
        
    end

end
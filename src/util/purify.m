function [AR, AR_vec] = purify(A)
    % Purify a quantum density matrix A

    N = length(A);

    if numel(A) == N
        
        % If A is a vector, assume A = diag(A)
        AR_vec = sparse(1:N+1:N^2, ones(N, 1), sqrt(A));
        AR = AR_vec * AR_vec';

    else

        % Otherwise compute purification normally
        [U, D] = eig(A);
        AR_vec = zeros(N^2, 1);
        for i = 1:N
            r_vec = zeros(N, 1); r_vec(i) = 1;
            AR_vec = AR_vec + sqrt(D(i, i)) * kron(U(:, i), r_vec);
        end
        AR = AR_vec * AR_vec';
        
    end

end
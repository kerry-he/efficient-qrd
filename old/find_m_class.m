clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))

lambda = 0.001:1:25.001;
n = 4;

%% Construct matrix


for L = 1:length(lambda)
    tic

    LAMBDA = lambda(L);
    a = exp(LAMBDA) / ( n^2 - n + n*exp(LAMBDA) );
    b = 1 / ( n^2 - n + n*exp(LAMBDA) );
    P = eye(n)*a + (ones(n) - eye(n))*b;
    
    % Compute mtx A
    H1 = diag(1./P(:));
    
    % Compute mtx B
    H2 = repmat({ones(n)}, 1, n);
    H2 = blkdiag(H2{:}) * n;

    
    v = zeros(n^2, n);
    for i = 1:n
        v(i:n:end, i) = 1;
    end
    [U_I, D] = eig(v*v');
    U_I = U_I(:, 1:end-n);


    H1u = U_I'*H1*U_I;
    H2u = U_I'*H2*U_I;

    P_I = (U_I * U_I');
    
    m(L) = 1 - eigs(H2u, H1u, 1, 'largestreal');

    fprintf("m = %.4f \t lambda = %.0f \n", m(L), LAMBDA)
    toc
end

m = real(m)';



%%

X = rand(n, n); Y = rand(n, n);
X = X ./ sum(X); Y = Y ./ sum(Y);
X = X / n; Y = Y / n;

x = X - Y;

sum(x.^2 ./ P, 'all') - sum( sum(x, 2).^2 ./ sum(P, 2) )



x_vec = x'; x_vec = x_vec(:);

x_vec' * (H1 - H2) * x_vec




%% Func

function [U, D] = fast_eig(X, P, idx_sp)
    N = sqrt(length(X));
    X = P*X*P';

    [U_sub, D_sub] = eig(full(X((N^2-N+1):end, (N^2-N+1):end)));

    D = zeros(N^2, 1);
    D(1:(N^2-N)) = diag(X(1:(N^2-N), 1:(N^2-N)));
    D((N^2-N+1):end) = diag(D_sub);
    
    v_sp = zeros(N^2-N + N^2, 1);
    v_sp(1:N^2-N) = 1;
    v_sp(N^2-N+1:end) = U_sub(:);
    U = sparse(idx_sp(:, 1), idx_sp(:, 2), v_sp);

    U = P'*U;
end
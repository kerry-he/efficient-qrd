clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))

lambda = 0.01:1:25.01;
n = 4;

a = 1 ./ (exp(lambda) + n^2 - 1);
b = 1 - n^2.*a;
m_lb = n^4 ./ min(1./b .* log(1+b./a), 1./(a+b) + 1./(a*(n^2-1)));

% Construct permutation matrix
sp_idx = zeros(n^2, 1);
alpha = 1; beta = n^2 - n + 1; count = 1;
for i = 1:n
    for j = 1:n
        if i ~= j
            sp_idx(count) = alpha;
            alpha = alpha + 1;
        else
            sp_idx(count) = beta;
            beta = beta + 1;
        end
        count = count + 1;
    end
end
P = sparse(sp_idx, 1:n^2, 1, n^2, n^2);

% Construct sparse indexing
k=n^2-n;
i_sp = zeros(k + n^2, 1); j_sp = i_sp; % Make sparse structure for U
i_sp(1:k) = 1:k; j_sp(1:k) = 1:k;
cnt = k + 1; M = n^2 - k;
for i = k+1:n^2
    i_sp(cnt:cnt+M-1) = k+1:n^2;
    j_sp(cnt:cnt+M-1) = i;
    cnt = cnt + M;
end
idx_sp = [i_sp, j_sp];


% Construct permutation matrix
sp_idx = zeros(n^4, 1);
alpha = 1; beta = n^4 - n^2 + 1; count = 1;
for i = 1:n^2
    for j = 1:n^2
        if i ~= j
            sp_idx(count) = alpha;
            alpha = alpha + 1;
        else
            sp_idx(count) = beta;
            beta = beta + 1;
        end
        count = count + 1;
    end
end
P2 = sparse(sp_idx, 1:n^4, 1, n^4, n^4);

% Construct sparse indexing
k=n^4-n^2;
i_sp = zeros(k + n^4, 1); j_sp = i_sp; % Make sparse structure for U
i_sp(1:k) = 1:k; j_sp(1:k) = 1:k;
cnt = k + 1; M = n^4 - k;
for i = k+1:n^4
    i_sp(cnt:cnt+M-1) = k+1:n^4;
    j_sp(cnt:cnt+M-1) = i;
    cnt = cnt + M;
end
idx_sp2 = [i_sp, j_sp];

%% Construct matrix

psi = kron(sparse(1, 1, 1, n, 1), sparse(1, 1, 1, n, 1));
for i = 2:n
    psi = psi + kron(sparse(i, 1, 1, n, 1), sparse(i, 1, 1, n, 1));
end

for L = 1:length(lambda)
    tic

    LAMBDA = lambda(L);
    a = 1 / ( exp(LAMBDA) + n^2 - 1 );
    b = (exp(LAMBDA) - 1) / ( exp(LAMBDA) + n^2 - 1 );
    rho = a*eye(n^2) + b*(psi*psi');
    
    [U, D] = fast_eig(rho, P, idx_sp);
    
    % Compute mtx A
    H1_diag = ones(n^2) / a;
    H1_diag(end, :) = (log(a+b) - log(a)) / b; H1_diag(:, end) = (log(a+b) - log(a)) / b;
    H1_diag(end, end) = 1 / (a+b);
    
    H1 = sparse(1:n^4, 1:n^4, H1_diag(:));
    
    H1 = kron(U, U) * H1 * kron(U, U)';
    
    
    % Compute mtx B
    idx = zeros(n^4, 1);
    jdx = zeros(n^4, 1);
    
    t = 0;
    for i = 1:n
        for j = 1:n
            for k = 1:n
                for l = 1:n
                    t = t + 1;
                    idx(t) = (((i - 1)*n + k) - 1)*n^2 + (j - 1)*n + k;
                    jdx(t) = (((i - 1)*n + l) - 1)*n^2 + (j - 1)*n + l;
                end
            end
        end
    end
    
    H2 = sparse(idx, jdx, n);

    t = 0;
    for k = 1:n
        for l = 1:n
            t = t + 1;
            for i = 1:n
                idx = (((i - 1)*n + k) - 1)*n^2 + (i - 1)*n + l;

                idx_mtx(idx, t) = 1;
            end
        end
    end

    [U_I, D_I] = eig(idx_mtx * idx_mtx');
    U_I = U_I(:, 1:end-n^2);

%     I = speye(n^2);
%     [U_I, D_I] = fast_eig(I(:) * I(:)', P2, idx_sp2);
%     U_I = U_I(:, 1:end-1);
    
    Pj = (U_I * U_I');
    
    H1u = U_I'*H1*U_I;
    H2u = U_I'*H2*U_I;
    
    
    m(L) = 1 - eigs(H2u, H1u, 1, 'largestreal');

    fprintf("m = %.4f \t lambda = %.0f \n", m(L), LAMBDA)
    toc
end

%%

H = rand(n^2) - 0.5; H = (H + H')/2;
H = reshape(Pj * H(:), [n^2, n^2]);

% H = [-1 0 0 0; 0 -1 0 0; 0 0 0 0; 0 0 0 0];

UHU = U'*H*U;
trR_H = PartialTrace(H);
vec_H = H(:);

f = 0;
for i = 1:n^2
    for j = 1:n^2
        f = f + H1_diag(i, j) * UHU(i, j)^2;
    end
end

for i = 1:n
    for j = 1:n
        f = f - n * trR_H(i, j)^2;
    end
end

f
vec_H' *Pj'* (H1 - H2) *Pj* vec_H

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

function fixed_rho = fix_R(rho, R)
    N = length(R);
    I = speye(N);
    fix = (R ./ diagPartialTrace(rho, 1, N)).^0.5;
    fix = sparse(1:N, 1:N, fix);

    fixed_rho = kron(I, fix) * rho * kron(I, fix');
end
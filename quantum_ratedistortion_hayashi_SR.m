clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))

rng(1)
N = 32; % Density matrix size

A = RandomDensityMatrix(N, 1);
[~, A] = eig(A);
AR_vec = sparse(purification(A));

AR = AR_vec * AR_vec'; 
AR = sparse(AR);
R = diagPartialTrace(AR, 1, N); R = sparse(1:N, 1:N, R);

% Construct permutation matrix
sp_idx = zeros(N^2, 1);
alpha = 1; beta = N^2 - N + 1; count = 1;
for i = 1:N
    for j = 1:N
        if i + j - N - 1 ~= 0
            sp_idx(count) = alpha;
            alpha = alpha + 1;
        else
            sp_idx(count) = beta;
            beta = beta + 1;
        end
        count = count + 1;
    end
end
P = sparse(sp_idx, 1:N^2, ones(N^2, 1), N^2, N^2);

% Construct sparse indexing
k=N^2-N;
i_sp = zeros(k + N^2, 1); j_sp = i_sp; % Make sparse structure for U
i_sp(1:k) = 1:k; j_sp(1:k) = 1:k;
cnt = k + 1; M = N^2 - k;
for i = k+1:N^2
    i_sp(cnt:cnt+M-1) = k+1:N^2;
    j_sp(cnt:cnt+M-1) = i;
    cnt = cnt + M;
end
idx_sp = [i_sp, j_sp];

%% Experiments
K = 500;
D = 0.0;
lambda = 3;

% Change of variables
RHO_QR = kron(A, R);
RHO_D = eig(RHO_QR);
V = logm(R);
DELTA = -AR;

S_A = von_neumann_entropy(A);

EPS = 1e-2;

k = 1;
while true
    tic
    
    RHO_Q = diagPartialTrace(RHO_QR, 2, N);
    obj(k) = S_A + shannon_entropy(RHO_Q); 
    obj(k) = obj(k) - shannon_entropy(RHO_D);
    lgn(k) = obj(k) - lambda*(AR_vec' * RHO_QR * AR_vec);

    % Perform Blahut-Arimoto iteration
    [RHO_QR, RHO_D, RHO_U, V] = trace_preserve_V(RHO_QR, V, lambda, AR, DELTA, P, idx_sp, EPS);

    t(k) = toc;

    % Compute objective value
    if k > 1
        EPS = max(min([abs(lgn(k) - lgn(k-1))^1, EPS]), 1e-15);
    end

    fprintf("Iteration: %d \t Objective: %.5e \t EPS: %.5e\n", k, obj(k), EPS)



    if k > 2
        if abs(lgn(k) - lgn(k-1)) < 1e-8
            break
        end
    end

    k = k + 1;
end

t_cum = cumsum(t');
fprintf("Elapsed time is %.6f seconds.\n", sum(t))

semilogy(lgn - lgn(end))

%% Functions

function S = von_neumann_entropy(rho)
    S = -trace(rho * logm(rho + eye(length(rho))*1e-10));
end

function [RHO, D, U, V] = trace_preserve_V(RHO, V, lambda, AR, DELTA, P, idx_sp, EPS)
    N = length(V);
    R = diagPartialTrace(AR, 1, N);
    B = diagPartialTrace(RHO, 2, N);

    LOG_BR = sparse(1:N^2, 1:N^2, log(kron(B, R)));

    [U, D] = fast_eig(LOG_BR + kron(speye(N), V) - lambda*DELTA, P, idx_sp);
    RHO = U * sparse(1:N^2, 1:N^2, exp(D)) * U';
    obj = log( trace( RHO ) ) - sum(diag(V) .* R);
    grad = diagPartialTrace(RHO, 1, N) / trace(RHO) - R;


    RHO_fix = fix_R(RHO, R);
    [U_fix, D_fix] = fast_eig(RHO_fix, P, idx_sp);

    gap = fast_quantum_relative_entropy(D_fix, U_fix, exp(D)/sum(exp(D)), U, P) - trace(RHO_fix) + trace(RHO/trace(RHO));



    fprintf("\t Error: %.5e\n", gap)

    
    alpha = 1e-8;
    beta = 0.1;
    while gap >= EPS
        
        t = 1000;
        while true

            V_next = V - t*diag(grad);
    
            [U, D] = fast_eig(LOG_BR + kron(speye(N), V_next) - lambda*DELTA, P, idx_sp);
            RHO = U * sparse(1:N^2, 1:N^2, exp(D)) * U';
            obj_next = log( trace( RHO ) ) - sum(diag(V_next) .* R);

            if obj_next <= obj - alpha*t*norm(grad)
                break
            end
            t = t * beta;

        end

        if obj - obj_next == 0
            break
        end

        grad = diagPartialTrace(RHO, 1, N) / trace(RHO) - R;
        obj = obj_next;
        V = V_next;

        RHO_fix = fix_R(RHO, R);
        [U_fix, D_fix] = fast_eig(RHO_fix, P, idx_sp);

        gap = fast_quantum_relative_entropy(D_fix, U_fix, exp(D)/sum(exp(D)), U, P) - trace(RHO_fix) + trace(RHO/trace(RHO));


        fprintf("\t Error: %.5e \t %.5e \t %.5e\n", gap, obj, EPS)

    end

    RHO = RHO / trace(RHO);
    D = exp(D); D = D / sum(D);
    
end

function out = diagPartialTrace(RHO, sys, N)
    % Faster partial trace operation when we know output is diagonal
    out = zeros(N, 1);
    if sys == 2
        for i = 1:N:N^2
            out((i-1)/N + 1) = sum(RHO(i:i+N-1, i:i+N-1), 'all');
        end
    else
        for i = 1:N
            out(i) = sum(RHO(i:N:N^2, i:N:N^2), 'all');
        end
    end
end

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

function H = shannon_entropy(p)
    p = p(p > 0);
    H = -p .* log(p);
    H = sum(H);
end

function H = relative_entropy(p, q)
    H = sum(p .* log(p ./ q));
end

function S = fast_quantum_relative_entropy(rho_D, rho_U, sigma_D, sigma_U, P)
    N = sqrt(length(rho_D));
    rho_U = P * rho_U;
    sigma_U = P * sigma_U;    

    S = relative_entropy(rho_D(1:(N^2-N)), sigma_D(1:(N^2-N)));

    rho_D = rho_D(N^2-N+1:end);
    sigma_D = sigma_D(N^2-N+1:end);

    rho_U = rho_U(N^2-N+1:end, N^2-N+1:end);
    sigma_U = sigma_U(N^2-N+1:end, N^2-N+1:end);

    rho_sigma_U = rho_U' * sigma_U;

    S = S - shannon_entropy(rho_D);
    S = S - trace(sparse(1:N, 1:N, rho_D) * rho_sigma_U * sparse(1:N, 1:N, log(sigma_D)) * rho_sigma_U');
end

function fixed_rho = fix_R(rho, R)
    N = length(R);
    I = speye(N);
    fix = (R ./ diagPartialTrace(rho, 1, N)).^0.5;
    fix = sparse(1:N, 1:N, fix);

    fixed_rho = kron(I, fix) * rho * kron(I, fix');
end

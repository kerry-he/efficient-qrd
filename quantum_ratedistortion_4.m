clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))

rng(2)
N = 32; % Density matrix size

A = RandomDensityMatrix(N, 1);
[~, A] = eig(A);
AR_vec = purification(A); AR = AR_vec * AR_vec'; AR = sparse(AR);
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
K = 100;
D = 0.0;
lambda = 8.75;

% Change of variables
RHO_QR = kron(A, R);
V_CV = logm(R);

RHO_QR_D = eig(RHO_QR);
RHO_QR_U = speye(N^2);
S_A = von_neumann_entropy(A);

tic
k = 1;
while true
    RHO_Q = diagPartialTrace(RHO_QR, 2, N);
    obj_cv(k) = S_A + shannon_entropy(RHO_Q); 
    obj_cv(k) = obj_cv(k) - shannon_entropy(RHO_QR_D);
    lgn_cv(k) = obj_cv(k) - lambda*(AR_vec' * RHO_QR * AR_vec);

    % Perform Blahut-Arimoto iteration
    [RHO_QR, RHO_QR_D, RHO_QR_U, V_CV] = trace_preserve_V(RHO_QR, V_CV, AR, lambda, P, idx_sp);

    % Compute upper bound
    grad = kron((log(RHO_Q) - log(diagPartialTrace(RHO_QR, 2, N))), ones(N, 1));
    grad = grad + kron(ones(N, 1), diag(V_CV));
    lb(k) = S_A;
    for i = 1:N
        lb(k) = lb(k) + min(grad(i:N:end)) * R(i, i);
    end    

    % Compute objective value
    fprintf("Iteration: %d \t Objective: %.5e \t DualGap: %.5e\n", k, obj_cv(k), lgn_cv(k) - lb(k))

    % Convert to bits
    obj_cv(k) = log2(exp(obj_cv(k)));
    lgn_cv(k) = log2(exp(lgn_cv(k)));
    lb(k) = log2(exp(lb(k)));

    if lgn_cv(k) - lb(k) < 1e-10
        break
    end

    k = k + 1;
end
toc

semilogy(obj_cv - obj_cv(end))
hold on
semilogy(lgn_cv - lb)

%% Functions

function S = von_neumann_entropy(rho)
    S = -trace(rho * logm(rho + eye(length(rho))*1e-10));
end

function H = shannon_entropy(p)
    p = p(p > 0);
    H = -p .* log(p);
    H = sum(H);
end

function [RHO_QR, rho_D, rho_U, V] = trace_preserve_V(RHO_QR, V, AR, lambda, P, idx_sp)
    NR = length(V);
    N = length(RHO_QR) / NR;
    R = diagPartialTrace(AR, 1, N);
    I = speye(N);

    % Cheap diagonalisation of X and V
    X_PREV = kron(sparse(1:N, 1:N, log(diagPartialTrace(RHO_QR, 2, N))), I);
    X = X_PREV + kron(I, V) + lambda*AR;
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

    % Function evaluation
    expD = exp(D);
    RHO_QR = U * sparse(1:N^2, 1:N^2, expD) * U';
    F = diagPartialTrace(RHO_QR, 1, N) - R;
%     F = diag(F);

    while norm(F) > 1e-15
        J = get_jacobian(D, U, idx_sp);

        % Compute Newton step
        p = J \ F;
    
        % Update with Newton step
        V = V - diag(p);
%         V = V - diag(F);

        % Cheap diagonalisation of X and V
        X = X_PREV + kron(I, V) + lambda*AR;
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

        % Function evaluation
        expD = exp(D);
        RHO_QR = U * sparse(1:N^2, 1:N^2, expD) * U';
        F_new = diagPartialTrace(RHO_QR, 1, N) - R;
%         F_new = diag(F_new);
        
        fprintf("\t Error: %.5e\n", norm(F_new))

        F = F_new;
    end

    rho_D = expD;
    rho_U = U;

end

function Df = exp_fdd(D, idx_sp)
    v = zeros(length(idx_sp), 1);

    for x = 1:length(idx_sp)
        i = idx_sp(x, 1); j = idx_sp(x, 2); 
        if i ~= j && D(i) ~= D(j)
            v(x) = (exp(D(i)) - exp(D(j))) / (D(i) - D(j));
        else
            v(x) = exp(D(i));
        end
    end

    Df = sparse(idx_sp(:, 1), idx_sp(:, 2), v);
end


function Df = fdd(Df, V, H)
    Df = Df .* (V'*H*V);
    Df = V * Df * V';
end

function [P, k, d, u] = deflation(d, u)
    N = length(u);

    % Check for ui = 0
    idx_z = find(abs(u) < eps);
    idx_nz = find(abs(u) >= eps);

    P = sparse(1:N, [idx_z; idx_nz], ones(N, 1));
    k = length(idx_z);

    d = P*d;
    u = P*u;
end

function [D_out, U_out] = dpr1(D_in, U_in, v, rho)
    % Finds eigenvalues and eigenvectors of a diagonalised plus rank 1
    % matrix A = U(D + rho*u*u^T)U^t

    N = length(v);

    % Sort diagonal matrix to be in ascending order
    [D_in, idx] = sort(diag(D_in));
    U_in = U_in(:, idx);

    % Rescale rank one update
    u = U_in' * v;
    u_norm = norm(u);
    rho = rho * u_norm^2;
    u = u / u_norm;

    % Deflation
    [P, k, D_in, u] = deflation(D_in, u);

    % Perform eigenvalue decomposition
    [D_out, ~, U_temp, ~] = laed9(D_in(k+1:end), u(k+1:end), rho);

    % Re-permute matrix from deflated state
    D_out = [D_in(1:k); D_out]; 
    D_out = P' * D_out;

    i_sp = zeros(k + (N - k)^2, 1); j_sp = i_sp; v_sp = i_sp; % Make sparse structure for U
    i_sp(1:k) = 1:k; j_sp(1:k) = 1:k; v_sp(1:k) = 1;
    cnt = k + 1; M = N - k;
    for i = k+1:N
        i_sp(cnt:cnt+M-1) = k+1:N;
        j_sp(cnt:cnt+M-1) = i;
        cnt = cnt + M;
    end
    v_sp(k+1:end) = U_temp(:);
    U_out = sparse(i_sp, j_sp, v_sp);

    U_out = P' * U_out * P;
    U_out = U_in * U_out;
end

function J = get_jacobian(D, U, idx_sp)
    N = sqrt(length(D));

    J = zeros(N, N);
    Df = exp_fdd(D, idx_sp);

    for i = 1:N
        H = sparse(i:N:N^2, i:N:N^2, 1, N^2, N^2);
        J_temp = fdd(Df, U, H);
        J(i, :) = diagPartialTrace(J_temp, 1, N);
    end
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
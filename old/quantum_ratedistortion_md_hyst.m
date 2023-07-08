clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))

rng(1)
N = 32; % Density matrix size

A = RandomDensityMatrix(N, 1);
[~, A] = eig(A);
AR_vec = sparse(purification(A));

% AR_vec(AR_vec ~= 0) = 1/sqrt(N);

AR = AR_vec * AR_vec'; 
AR = sparse(AR);
R = diagPartialTrace(AR, 1, N); R = sparse(1:N, 1:N, R);

I = speye(N);

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
% list_lambda = 12.5:-0.005:0.1;
list_lambda = 0.1:0.005:12.5;


% Change of variables
% RHO_QR = kron(sparse(N, N, 1), R);
RHO_QR = kron(A, R);
% RHO_QR = AR;

RHO_inf = RHO_QR;
V = -logm(R);

RHO_QR_D = eig(RHO_QR);
RHO_QR_U = speye(N^2);
S_A = von_neumann_entropy(A);

y_idx = zeros(N, 1);


for j = 1:length(list_lambda)
    lambda = list_lambda(j);
    obj = [];
    lgn = [];
    lb = [];
    k = 1;
    EPS = 1e-2;

%     RHO_QR = kron(A, R);
%     
%     RHO_inf = RHO_QR;
%     V = -logm(R);
    
%     RHO_QR_D = eig(RHO_QR);
%     RHO_QR_U = speye(N^2);

    tic
    while true
        
        RHO_Q = diagPartialTrace(RHO_QR, 2, N);
        obj(k) = S_A + shannon_entropy(RHO_Q); 
        obj(k) = obj(k) - shannon_entropy(RHO_QR_D);
        lgn(k) = obj(k) - lambda*(AR_vec' * RHO_QR * AR_vec);
    
        % Perform Blahut-Arimoto iteration
        [RHO_QR, RHO_QR_D, RHO_QR_U, V, RHO_inf] = trace_preserve_V(RHO_inf, V, AR, lambda, P, idx_sp, EPS);
    
        % Compute upper bound
        grad = kron((log(RHO_Q) - log(diagPartialTrace(RHO_QR, 2, N))), ones(N, 1));
        grad = grad - kron(ones(N, 1), diag(V));
        lb(k) = S_A;
        for i = 1:N
            lb(k) = lb(k) + min(grad(i:N:end)) * R(i, i);
        end
    
        % Compute objective value
        if k > 1
            EPS = max(min([abs(lgn(k) - lgn(k-1))^1, EPS]), 1e-15);
        end
%         fprintf("Iteration: %d \t Objective: %.5e \t DualGap: %.5e\t EPS: %.5e\n", k, obj(k), lgn(k) - lb(k), EPS)
    
        % Convert to bits
        
    
        if k > 2
            if abs(lgn(k) - lgn(k-1)) < 1e-8
                break
            end
        end
    
        k = k + 1;
    end

    t(j) = toc;
    fprintf("lambda=%.2f\t  t=%.6f seconds.\n", lambda, t(j))

    rate(j) = obj(end);
    distortion(j) = 1 - AR_vec' * RHO_QR * AR_vec;

end

fprintf("Total time elapsed: %.6f seconds.\n", sum(t))

%% Functions

function S = von_neumann_entropy(rho)
    S = -trace(rho * logm(rho + eye(length(rho))*1e-10));
end

function H = shannon_entropy(p)
    p = p(p > 0);
    H = -p .* log(p);
    H = sum(H);
end

function fixed_rho = fix_R(rho, R)
    N = length(R);
    I = speye(N);
    fix = (R ./ diagPartialTrace(rho, 1, N)).^0.5;
    fix = sparse(1:N, 1:N, fix);

    fixed_rho = kron(I, fix) * rho * kron(I, fix');
end

function [RHO_QR, rho_D, rho_U, V, RHO_inf] = trace_preserve_V(RHO_QR, V, AR, lambda, P, idx_sp, EPS)
    NR = length(V);
    N = length(RHO_QR) / NR;
    R = diagPartialTrace(AR, 1, N);
    I = speye(N);

    % Cheap diagonalisation of X and V
    X_PREV = kron(sparse(1:N, 1:N, log(diagPartialTrace(RHO_QR, 2, N))), I);
    X = X_PREV - kron(I, V) + lambda*AR;
    [U, D] = fast_eig(X, P, idx_sp);

    % Function evaluation
    expD = exp(D);
    RHO_QR = U * sparse(1:N^2, 1:N^2, expD) * U';
    obj = -trace(RHO_QR) - sum(R .* diag(V));
    F = diagPartialTrace(RHO_QR, 1, N) - R;

    RHO_QR_fix = fix_R(RHO_QR, R);
    [U_fix, D_fix] = fast_eig(RHO_QR_fix, P, idx_sp);

    gap = fast_quantum_relative_entropy(D_fix, U_fix, expD, U, P) - trace(RHO_QR_fix) + trace(RHO_QR);
    
%     fprintf("\t Error: %.5e\n", gap)

    alpha = 1e-7;
    beta = 0.1;
    while gap >= EPS

        % Compute Newton step
        J = get_jacobian(D, U, idx_sp);
        p = J \ F;

        t = 1;
        while true
            % Update with Newton step
%             step = diag(F);
            step = -diag(p);
            V_new = V + t*step;
    
            % Cheap diagonalisation of X and V
            X = X_PREV - kron(I, V_new) + lambda*AR;
            [U, D] = fast_eig(X, P, idx_sp);
    
            % Function evaluation
            expD = exp(D);
            RHO_QR = U * sparse(1:N^2, 1:N^2, expD) * U';
            obj_new = -trace(RHO_QR) - sum(R .* diag(V_new));

            if -obj_new <= -obj - alpha*t*trace(diag(F) * step)
                break
            end
            t = t * beta;
        end

        if obj - obj_new == 0
            break
        end

        F = diagPartialTrace(RHO_QR, 1, N) - R;
        obj = obj_new;
        V = V_new;

        RHO_QR_fix = fix_R(RHO_QR, R);
        [U_fix, D_fix] = fast_eig(RHO_QR_fix, P, idx_sp);

        gap = fast_quantum_relative_entropy(D_fix, U_fix, expD, U, P) - trace(RHO_QR_fix) + trace(RHO_QR);
    
%         fprintf("\t Error: %.5e\n", gap)
    end

    RHO_inf = RHO_QR;
    RHO_QR = RHO_QR_fix;
    rho_D = D_fix;
    rho_U = U_fix;

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
        H = sparse(i:N:N^2, i:N:N^2, -1, N^2, N^2);
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

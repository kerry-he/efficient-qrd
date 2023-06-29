clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))

rng(1)
N = 2; % Density matrix size

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
K = 10000;
DIST = 0.5;
lambda = 3;
lambda_prev = lambda;

% Change of variables
RHO_QR = kron(A, R);
V_CV = -logm(R);
V_prev = V_CV;

RHO_QR_D = eig(RHO_QR);
RHO_QR_U = speye(N^2);
S_A = von_neumann_entropy(A);

RHO_Q = diagPartialTrace(RHO_QR, 2, N);
obj(1) = S_A + shannon_entropy(RHO_Q); 
obj(1) = obj(1) - shannon_entropy(RHO_QR_D);
lgn(1) = obj(1) - lambda*(AR_vec' * RHO_QR * AR_vec);

t_p = 1;
t_d = 10;
theta_0 = 1.0;

tic
k = 2;
while true
    tic
    RHO_QR_prev = RHO_QR;
    RHO_QR_D_prev = RHO_QR_D;
    RHO_QR_U_prev = RHO_QR_U;    
    V_prev_prev = V_prev;
    V_prev = V_CV;
    lambda_prev_prev = lambda_prev;
    lambda_prev = lambda;

    theta = 1.01;
    t_p = t_p * theta;
    t_d = t_d * theta;

    while true

        % Perform Blahut-Arimoto iteration
        V_bar = V_prev + theta*(V_prev - V_prev_prev);
        lambda_bar = lambda_prev + theta*(lambda_prev_prev - lambda_prev_prev);

        LOG_RHO_QR = RHO_QR_U_prev * sparse(1:N^2, 1:N^2, log(RHO_QR_D_prev)) * RHO_QR_U_prev';
        GRAD_RHO = LOG_RHO_QR - kron(sparse(1:N, 1:N, log(diagPartialTrace(RHO_QR_prev, 2, N))), eye(N));
        STEP = -GRAD_RHO + lambda_bar*AR - kron(speye(N), V_bar);
        
        [U, D] = fast_eig(LOG_RHO_QR + STEP*t_p, P, idx_sp);

        RHO_QR_D = exp(D);
        RHO_QR_D = RHO_QR_D / sum(RHO_QR_D);
        RHO_QR_U = U;
        RHO_QR = RHO_QR_U * sparse(1:N^2, 1:N^2, RHO_QR_D) * RHO_QR_U';
        
        V_CV = V_prev + (diag(diagPartialTrace(RHO_QR, 1, N)) - R) * t_d;
        lambda = lambda_prev + t_d * (1 - (AR_vec' * RHO_QR * AR_vec) - DIST)*0;

        RHO_Q = diagPartialTrace(RHO_QR, 2, N);
        obj(k) = S_A + shannon_entropy(RHO_Q); 
        obj(k) = obj(k) - shannon_entropy(RHO_QR_D);

        if obj(k) - obj(k - 1) - trace(GRAD_RHO * (RHO_QR - RHO_QR_prev)) ...
            + (diag(V_CV) - diag(V_bar))' * diagPartialTrace(RHO_QR - RHO_QR_prev, 1, N) - (lambda-lambda_bar)*(AR_vec' * RHO_QR * AR_vec - (AR_vec' * RHO_QR_prev * AR_vec)) ...
                <= abs(ffast_quantum_relative_entropy(RHO_QR_D, RHO_QR_U, RHO_QR_D_prev, RHO_QR_U_prev, P) - trace(RHO_QR) + trace(RHO_QR_prev)) / t_p...
                + (norm(V_CV - V_bar, 'fro')^2 + (lambda-lambda_bar)^2) / (2*t_d)
            break
        end

        theta = theta * 0.75;
        t_p = t_p * 0.75;
        t_d = t_d * 0.75;

        display("BACKTRACKING")

    end


    % Compute objective value
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj(k))

    % Convert to bits
%     obj_cv(k) = log2(exp(obj_cv(k)));
%     lgn_cv(k) = log2(exp(lgn_cv(k)));
    lgn(k) = obj(k) - lambda*(AR_vec' * RHO_QR * AR_vec);
    time(k) = toc;

%     d_res(k) = (norm(lambda - lambda_prev)^2 + norm(V_CV - V_prev, 'fro')^2) / (2*t_d*max([1, lambda, max(abs(V_CV), [], 'all')]));
%     p_res(k) = (ffast_quantum_relative_entropy(RHO_QR_D, RHO_QR_U, RHO_QR_D_prev, RHO_QR_U_prev, P) - trace(RHO_QR) + trace(RHO_QR_prev)) / (t_p);
% 
%     if p_res(k) + d_res(k) <= 1e-10 && k > 10
%         break
%     end    

    if k > 2
        if abs(lgn(k) - lgn(k-1)) < 1e-8
            break
        end
    end

    k = k + 1;
end

fprintf("Elapsed time is %.6f seconds.\n", sum(time))

semilogy(cumsum(time(1:end)), abs(obj - obj(end)))

%% Functions

function S = von_neumann_entropy(rho)
    S = -trace(rho * logm(rho + eye(length(rho))*1e-10));
end

function S = quantum_relative_entropy(rho, sigma)
    S = trace(rho * (logm(rho) - logm(sigma)));
end

function S = fast_quantum_relative_entropy(rho_D, rho_U, sigma_D, sigma_U)
    N = length(rho_D);
    log_rho = rho_U * sparse(1:N, 1:N, log(rho_D)) * rho_U';
    log_sigma = sigma_U * sparse(1:N, 1:N, log(sigma_D)) * sigma_U';

    S = trace(rho_U * sparse(1:N, 1:N, log(rho_D)) * rho_U * (log_rho - log_sigma));
end

function S = ffast_quantum_relative_entropy(rho_D, rho_U, sigma_D, sigma_U, P)
    N = sqrt(length(rho_D));
%     rho_D = P * rho_D;
    rho_U = P * rho_U;
%     sigma_D = P * sigma_D;
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

function H = shannon_entropy(p)
    p = p(p > 0);
    H = -p .* log(p);
    H = sum(H);
end

function H = relative_entropy(p, q)
    H = sum(p .* log(p ./ q));
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

    while true
        J = get_jacobian(D, U, idx_sp);

        % Compute Newton step
        p = J \ F;

        newton_dec = F' * p / 2;
        if newton_dec < 1e-25
            fprintf("\t Error: %.5e \t %.5e\n", norm(F), newton_dec)
            break
        end
    
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
        
        fprintf("\t Error: %.5e \t %.5e\n", norm(F_new), newton_dec)

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
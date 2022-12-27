clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))

rng(2)
N = 64; % Density matrix size

A = RandomDensityMatrix(N, 1);
[~, A] = eig(A);
AR_vec = purification(A); AR = AR_vec * AR_vec';

%% Experiments
K = 20;
D = 0.0;
lambda = 5.0;

% Change of variables
% RHO_QR = RandomDensityMatrix(N*NR, 1);
RHO_QR = kron(A, PartialTrace(AR, 1));
V_CV = -logm(PartialTrace(AR, 1));
% V_CV = zeros(N, N);
F_grad = AR;
dF = eye(N^2, N^2);
% dF = [];

RHO_QR_eig = eig(RHO_QR);
S_A = von_neumann_entropy(A);

tic
for k = 1:K
    obj_cv(k) = S_A + von_neumann_entropy(PartialTrace(RHO_QR, 2, N)); 
    obj_cv(k) = obj_cv(k) - shannon_entropy(RHO_QR_eig);
    lgn_cv(k) = obj_cv(k) + lambda*(1 - AR_vec' * RHO_QR * AR_vec);

    % Perform Blahut-Arimoto iteration
    [RHO_QR, RHO_QR_eig, V_CV, dF] = trace_preserve_V(RHO_QR, V_CV, AR_vec, lambda, dF);

    % Compute objective value
    fprintf("Iteration: %d \t Objective: %.5e \t lambda: %.5f\n", k, obj_cv(k), lambda)
end
toc

semilogy(obj_cv - obj_cv(end))

%% Functions

function S = von_neumann_entropy(rho)
    S = -trace(rho * logm(rho + eye(length(rho))*1e-10));
end

function H = shannon_entropy(p)
    p = p(p > 0);
    H = -p .* log(p);
    H = sum(H);
end

function [RHO_QR, expD, V, dF] = trace_preserve_V(RHO_QR, V, AR, lambda, dF)
    NR = length(V);
    N = length(RHO_QR) / NR;
    R = PartialTrace(AR*AR', 1, N);
    I = eye(N);

    % Cheap diagonalisation of X and V
    [Ux, Dx] = eig(full(logm(PartialTrace(RHO_QR, 2, N))));
    [Uv, Dv] = eig(V);
    Ux = sparse(Ux); Uv = sparse(Uv);
%     Uv = eye(N); Dv = V;

    Dxv = kron(Dx, I) - kron(I, Dv);
    Uxv = kron(Ux, Uv);

    [D, U] = dpr1(Dxv, Uxv, AR, lambda);

    % Function evaluation
    expD = exp(D);
    RHO_QR = U * sparse(diag(expD)) * U';
    F = PartialTrace(RHO_QR, 1, N) - R;
    F = F(:);

    while norm(F) > 1e-10
        Df = exp_fdd(D);
        dFdH = @(H)reshape(PartialTrace(fdd(Df, U, -kron(I, reshape(H', [N, N]))), 1, N), [N^2, 1]);

        % Compute step direction with Krylov method
        [p] = minres(dFdH, F, 1e-2, 20, diag(1./abs(diag(dF))));
    
        % Update with Newton step
        V_vec = V(:) - p;
        V = reshape(V_vec, size(V));

        % Cheap diagonalisation of X and V
        [Uv, Dv] = eig(V);
        Uv = sparse(Uv);

        Dxv = kron(Dx, I) - kron(I, Dv);
        Uxv = kron(Ux, Uv);

        [D, U] = dpr1(Dxv, Uxv, AR, lambda);

        % Function evaluation
        expD = exp(D);
        RHO_QR = U * sparse(diag(expD)) * U';
        F_new = PartialTrace(RHO_QR, 1, N) - R;
        F_new = F_new(:);
        
        s = -p;
        y = F_new - F;
        eta = 1 / (y'*s);
        dF = (eye(NR^2) - eta*s*y') * dF * (eye(NR^2) - eta*y*s') + eta*(s*s');
        fprintf("\t Error: %.5e\n", norm(F_new))

        F = F_new;
    end

end

function Df = exp_fdd(D)
    N = length(D);
    Df = zeros(N, N);
    for i = 1:N
        for j = 1:i
            if i ~= j && D(i) ~= D(j)
                Df(i, j) = (exp(D(i)) - exp(D(j))) / (D(i) - D(j));
            else
                Df(i, j) = exp(D(i));
            end
        end
    end

    Df = Df + tril(Df, -1)';
end


function Df = fdd(Df, V, H)
    H = sparse(H);
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

%     U_out = eye(N);
%     U_out(k+1:end, k+1:end) = U_temp;

    U_out = P' * U_out * P;
    U_out = U_in * U_out;
end
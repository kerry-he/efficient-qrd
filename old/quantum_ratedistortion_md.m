clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))

rng(1)
N = 128; % Density matrix size

A = RandomDensityMatrix(N, 1);
% [~, A] = eig(A);
AR_vec = purification(A); 
AR = AR_vec * AR_vec'; 
R = PartialTrace(AR, 1, N);


%% Experiments
K = 100;
D = 0.5;
lambda = 7;

% Change of variables
RHO_QR = kron(A, R);
% diag0 = rand(N^2, 1); diag0 = diag0 / sum(diag0);
% RHO_QR = sparse(1:N^2, 1:N^2, diag0);


RHO_inf = RHO_QR;
V = -logm(R);

RHO_QR_D = eig(RHO_QR);
RHO_QR_U = speye(N^2);
S_A = von_neumann_entropy(A);

y_idx = zeros(N, 1);

EPS = 1e-2;

exitflag = false;
k = 1;
while true
    tic
    obj(k) = S_A + von_neumann_entropy(PartialTrace(RHO_QR, 2, N)) - von_neumann_entropy(RHO_QR);
    lgn(k) = obj(k) - lambda*(AR_vec' * RHO_QR * AR_vec);

    % Perform Blahut-Arimoto iteration
    [RHO_QR, V, RHO_inf] = trace_preserve_V(RHO_inf, V, AR, lambda, EPS);

    % Compute objective value

    t(k) = toc;

    if k > 1
        EPS = max(min([abs(lgn(k) - lgn(k-1))^1, EPS]), 1e-15);
        fprintf("Iteration: %d \t Objective: %.5e \t DualGap: %.5e\t EPS: %.5e\n", k, obj(k), lgn(k) - lgn(k-1), EPS)
    end

    % Convert to bits
%     obj_cv(k) = log2(exp(obj_cv(k)));
%     lgn_cv(k) = log2(exp(lgn_cv(k)));
%     lb(k) = log2(exp(lb(k)));
    dual(k) = lambda;


    if k > 2
        if abs(lgn(k) - lgn(k-1)) < 1e-8
            break
        end
    end

    k = k + 1;
end

t_cum = cumsum(t');
fprintf("Elapsed time is %.6f seconds.\n", sum(t))


semilogy(obj - obj(end))
hold on


% cvx_begin sdp
%     variable rho(N^2, N^2) symmetric
%     minimize (quantum_entr(A) - quantum_cond_entr(rho, [N, N], 2) - lambda * trace(rho * AR));
%     rho >= 0;
%     PartialTrace(rho, 1, N) == R;
% cvx_end
% cvx_time = cvx_cputime;
% cvx_opt = cvx_optval;

% RHO_QR = full(RHO_QR);
% grad_new = logm(RHO_QR) - kron(logm(PartialTrace(RHO_QR)), eye(N)) - AR;
% 
% cvx_begin sdp
%     variable rho(N^2, N^2) symmetric
%     minimize ( trace( rho * grad_new ) );
%     rho >= 0;
%     PartialTrace(rho, 1, N) == R;
% cvx_end


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
    fix = (R * PartialTrace(rho, 1, N)^(-1)).^0.5;
    fix = real(fix);
    fixed_rho = kron(I, fix) * rho * kron(I, fix');
end

function [RHO_QR, V, RHO_inf] = trace_preserve_V(RHO_QR, V, AR, lambda, EPS)
    NR = length(V);
    N = length(RHO_QR) / NR;
    R = PartialTrace(AR, 1, N);
    I = speye(N);

    % Cheap diagonalisation of X and V
    X_PREV = kron(logm(PartialTrace(RHO_QR, 2, N)), I);
    X = X_PREV - kron(I, V) + lambda*AR;
    [U, D] = eig(X);

    % Function evaluation
    RHO_QR = U * diag(exp(diag(D))) * U';
    obj = -trace(RHO_QR) - trace(R * V);
    F = PartialTrace(RHO_QR, 1, N) - R;

    RHO_QR_fix = fix_R(RHO_QR, R);
    gap = trace(RHO_QR_fix * (logm(RHO_QR_fix) - logm(RHO_QR))) - trace(RHO_QR_fix) + trace(RHO_QR);
    
    fprintf("\t Error: %.5e\n", gap)

    alpha = 1e-7;
    beta = 0.1;
    while gap >= EPS

        % Compute Newton step
%         J = get_jacobian(D, U);
%         p = reshape(J \ F(:), [N, N]);

        t = 1000;
        while true
            % Update with Newton step
            step = F;
%             step = -p;
            V_new = V + t*step;
    
            % Cheap diagonalisation of X and V
            X = X_PREV - kron(I, V_new) + lambda*AR;
            [U, D] = eig(X);
        
            % Function evaluation
            RHO_QR = U * diag(exp(diag(D))) * U';
            obj_new = -trace(RHO_QR) - trace(R * V_new);

            if -obj_new <= -obj - alpha*t*trace(F * step)
                break
            end
            t = t * beta;
        end

        if obj - obj_new == 0
            break
        end

        F = PartialTrace(RHO_QR, 1, N) - R;
        obj = obj_new;
        V = V_new;

        RHO_QR_fix = fix_R(RHO_QR, R);
        gap = trace(RHO_QR_fix * (logm(RHO_QR_fix) - logm(RHO_QR))) - trace(RHO_QR_fix) + trace(RHO_QR);
    
        fprintf("\t Error: %.5e\n", gap)
    end

    RHO_inf = RHO_QR;
    RHO_QR = RHO_QR_fix;

end

function Df = exp_fdd(D)
    N = length(D);
    Df = zeros(N, N);

    for i = 1:N
        for j = 1:N
            if i == j || D(i, i) == D(j, j)
                Df(i, j) = exp(D(i, j));
            else
                Df(i, j) = (exp(D(i, i)) - exp(D(j, j))) / (D(i, i) - D(j, j));
            end
        end
    end
end

function Df = fdd(Df, V, H)
    Df = Df .* (V'*H*V);
    Df = V * Df * V';
end

function J = get_jacobian(D, U)
    N = sqrt(length(D));

    J = zeros(N^2, N^2);
    Df = exp_fdd(D);

    for i = 1:N
        for j = 1:N
            if i > j
                temp = reshape(J((j - 1)*N + i, :), [N, N]);
                temp = temp';
                J((i - 1)*N + j, :) = temp(:);
            else
                H = sparse(i, j, 1, N, N);
                dFdH = PartialTrace(fdd(Df, U, -kron(eye(N), H)), 1, N);        
                J((i - 1)*N + j, :) = dFdH(:);
            end
        end
    end    
end

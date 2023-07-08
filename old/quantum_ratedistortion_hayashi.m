clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))

rng(1)
N = 8; % Density matrix size

A = RandomDensityMatrix(N, 1);
AR_vec = purification(A); AR = AR_vec * AR_vec';



%% Experiments
K = 500;
D = 0.0;
lambda = 1;

% Change of variables
% RHO_QR = RandomDensityMatrix(N*NR, 1);
RHO_QR = kron(A, PartialTrace(AR, 1));
V = logm(PartialTrace(AR, 1));
DELTA = -AR;

S_A = von_neumann_entropy(A);
EPS = 1e-2;

for k = 1:K
    tic

    obj(k) = S_A + von_neumann_entropy(PartialTrace(RHO_QR, 2, N)) - von_neumann_entropy(RHO_QR);
    lgn(k) = obj(k) - lambda*(AR_vec' * RHO_QR * AR_vec);

    % Perform Blahut-Arimoto iteration
    [RHO_QR, V] = trace_preserve_V(RHO_QR, V, lambda, AR, DELTA, EPS);

    % Compute objective value
    if k > 1
        EPS = max(min([abs(lgn(k) - lgn(k-1))^1, EPS]), 1e-15);
    end
    
    fprintf("Iteration: %d \t Objective: %.5e \t lambda: %.5f\n", k, obj(k), lambda)

    t(k) = toc;

    if k > 2
        if abs(lgn(k) - lgn(k-1)) < 1e-8
            break
        end
    end
end

t_cum = cumsum(t');
fprintf("Elapsed time is %.6f seconds.\n", sum(t))

semilogy(lgn - lgn(end))

%% Functions

function S = von_neumann_entropy(rho)
    S = -trace(rho * logm(rho + eye(length(rho))*1e-10));
end

function [RHO, V] = trace_preserve_V(RHO, V, lambda, AR, DELTA, EPS)
    N = length(V);
    R = PartialTrace(AR, 1, N);
    B = PartialTrace(RHO, 2, N);

    RHO = expm( logm(kron(B, R)) + kron(eye(N), V) - lambda*DELTA );
    obj = log( trace( RHO ) ) - trace(V * R);
    grad = PartialTrace(RHO, 1, N) / trace(RHO) - R;

    RHO_fix = fix_R(RHO, R);
    gap = trace(RHO_fix * (logm(RHO_fix) - logm(RHO/trace(RHO)))) - trace(RHO_fix) + trace(RHO/trace(RHO));


    fprintf("\t Error: %.5e\n", gap)
    
    alpha = 1e-8;
    beta = 0.1;
    while gap >= EPS
        
        t = 1000;
        while true

            V_next = V - t*grad;
    
            RHO = expm( logm(kron(B, R)) + kron(eye(N), V_next) - lambda*DELTA );
            obj_next = log( trace( RHO ) ) - trace(V_next * R);
            

            if obj_next <= obj - alpha*t*norm(grad, 'fro')
                break
            end
            t = t * beta;

        end

        grad = PartialTrace(RHO, 1, N) / trace(RHO) - R;
        obj = obj_next;
        V = V_next;        

        RHO_fix = fix_R(RHO, R);
        gap = trace(RHO_fix * (logm(RHO_fix) - logm(RHO/trace(RHO)))) - trace(RHO_fix) + trace(RHO/trace(RHO));
        fprintf("\t Error: %.5e \t %.5e\n", gap, obj)

    end

    RHO = RHO / trace(RHO);
    
end

function fixed_rho = fix_R(rho, R)
    N = length(R);
    I = speye(N);
    fix = (R * PartialTrace(rho, 1, N)^(-1)).^0.5;

    fixed_rho = kron(I, fix) * rho * kron(I, fix');
end
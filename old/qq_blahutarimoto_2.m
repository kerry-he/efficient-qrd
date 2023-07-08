clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))

rng(6)
N = 64; % Density matrix size
M = 32;

rho = ones(N, N) / N;
rho = rand_SPD(N); rho = rho / trace(rho);

V = randV(N*M, N, 1);

% PHI = KrausOperators(RandomSuperoperator([N, N], 1, 1, 1));
% % PHI = KrausOperators(DepolarizingChannel(N, 0.25));
% PHI_C = ComplementaryMap(PHI, [N, N]);
% 
% PHI_adj = PHI;
% for i = 1:length(PHI_adj)
%     PHI_adj{i} = PHI_adj{i}';
% end
% PHI_C_adj = PHI_C;
% for i = 1:length(PHI_C_adj)
%     PHI_C_adj{i} = PHI_C_adj{i}';
% end

Nc = 6; % Number of constraints
for i = 1:Nc
    A{i} = RandomDensityMatrix(N, 1)*N;
    b(i) = rand;
end

% cvx_begin sdp
%     variable X(N, N) symmetric
% 
%     maximize ( 0 );
% 
%     X >= 0;
%     trace(X) == 1;
%     for i = 1:Nc
%         trace(A{i}*X) <= b(i);
%     end
% cvx_end

display("DONE")


%% Algorithm
% Primal and dual variables
rho = rand_SPD(N); rho = rho / trace(rho);
lambda = zeros(Nc, 1);
lambda_prev = lambda;

t_p = 5;
t_d = 50;

obj(1) = q_mutual_inf(rho, V, [N, M]);

for k = 1:1000
    tic
    % Step sizes
    theta = 1.01;
    t_p = t_p*theta;
    t_d = t_d*theta;

    lambda_prev_prev = lambda_prev;

    rho_prev = rho;
    lambda_prev = lambda;

    [~, grad] = q_mutual_inf(rho, V, [N, M], "grad");
    

    while true
        lambda_bar = lambda_prev + theta*(lambda_prev - lambda_prev_prev);
        
        grad_dual = zeros(N);
        for i = 1:Nc
            grad_dual = grad_dual + lambda_bar(i) * A{i};
        end
        rho = expm(logm(rho_prev) + t_p * (grad - grad_dual));
        rho = rho / trace(rho);

        for i = 1:Nc
            lambda(i) = lambda_prev(i) + t_d * (trace(A{i}*rho) - b(i)); 
        end
        lambda = max(0, real(lambda));

        % Compute gradients
        obj(k+1) = q_mutual_inf(rho, V, [N, M], 'func');
        obj_dual = zeros(Nc, 1);
        for i = 1:Nc
            obj_dual = obj_dual + trace(A{i} * (rho - rho_prev));
        end

        if -obj(k+1) <= -obj(k) - trace(grad*(rho - rho_prev)) + (quantum_relative_entropy(rho, rho_prev) - trace(rho) + trace(rho_prev)) / t_p ...
                + sum((lambda-lambda_bar).^2, 'all')/(2*t_d) - ((lambda-lambda_bar)' * obj_dual)
            break
        end

        fprintf("BACKTRACKING\n")
        theta = theta * 0.75;
        t_p = t_p*theta;
        t_d = t_d*theta;
    end

    time(k) = toc;
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj(k))

    p_res(k) = real(norm(lambda - lambda_prev) / (t_d * max(1, max(abs(lambda)))));
    d_res(k) = real(quantum_relative_entropy(rho, rho_prev) / (t_p));
% 
%     if p_res(k) + d_res(k) <= 1e-7
%         break
%     end        

end

fprintf("Elapsed time is %.6f seconds.\n", sum(time))

%% Backtracking MD
rho = rand_SPD(N); rho = rho / trace(rho);
t = 1;

obj_md(1) = q_mutual_inf(rho, V, [N, M], "grad");

for k = 1:50
    tic

    if k > 1
        grad_prev = grad;
    end
    rho_prev = rho;

    [~, grad] = q_mutual_inf(rho, V, [N, M], "grad");
    
    while true

        cvx_begin sdp quiet
            variable X(N, N) symmetric
        
            minimize ( trace(-t*grad * X) + quantum_rel_entr(X, rho_prev) );
        
            X >= 0;
            trace(X) == 1;
            for i = 1:Nc
                trace(A{i}*X) <= b(i);
            end
        cvx_end        
        
        rho = X;
        
        obj_md(k + 1) = q_mutual_inf(rho, V, [N, M], 'func');

        if t < 1 || -obj_md(k + 1) <= -obj_md(k) - trace(grad*(rho - rho_prev)) + 1/t * quantum_relative_entropy(rho, rho_prev)
            break
        end

        t = t / 2;
    end

    if k > 2
        t = quantum_relative_entropy(rho, rho_prev) / trace(rho * (grad_prev - grad));
        if isinf(t) || isnan(t) || t < 0
            t = 1;
        end
    end

    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_md(k+1))
    time_md(k) = toc;
end

%% CVX
cvx_solver_settings('maxit', 100)
tic
cvx_begin sdp
    variable X(N, N) symmetric

    maximize ( quantum_cond_entr(V*X*V', [N M]) + quantum_entr(TrX(V*X*V',2,[N M])) );

    X >= 0;
    trace(X) == 1;
    for i = 1:Nc
        trace(A{i}*X) <= b(i);
    end
cvx_end
toc

figure
semilogy(cumsum(time), abs(obj(2:end) - cvx_optval))
hold on
semilogy(cumsum(time_md), abs(obj_md(2:end) - cvx_optval))


%% Functions
function S = von_neumann_entropy(rho)
    S = -trace(rho * logm(rho));
end

function S = quantum_relative_entropy(rho, sigma)
    S = trace(rho * (logm(rho) - logm(sigma)));
end

function out = holevo_inf(p, rho)
    [N, ~, n] = size(rho);

    Rho = zeros(N, N);
    for i = 1:n
        Rho = Rho + p(i)*rho(:, :, i);
    end

    out = von_neumann_entropy(Rho);
    for i = 1:n
        out = out - p(i)*von_neumann_entropy(rho(:, :, i));
    end
end


function [func, grad] = q_mutual_inf(rho, V, dim, type)
    PHI_rho = ApplyStinespring(rho, V, dim, 0, 0);
    PHI_C_rho = ApplyStinespring(rho, V, dim, 1, 0);

    if ~exist('type','var')
        type = "";
    end

    grad = 0;
    func = 0;
    
    if type == "func" || type == ""
        func = von_neumann_entropy(rho);
        func = func + von_neumann_entropy(PHI_rho);
        func = func - von_neumann_entropy(PHI_C_rho);
    end

    if type == "grad" || type == ""
        grad = -logm(rho);
        grad = grad - ApplyStinespring(logm(PHI_rho), V, dim, 0, 1);
        grad = grad + ApplyStinespring(logm(PHI_C_rho), V, dim, 1, 1);
        grad = grad - eye(length(grad));
    end

end


function [func, grad] = q_coherent_inf(rho, PHI, PHI_adj, PHI_C, PHI_C_adj)
    PHI_rho = ApplyMap(rho, PHI);
    PHI_C_rho = ApplyMap(rho, PHI_C);

    func = von_neumann_entropy(PHI_rho);
    func = func - von_neumann_entropy(PHI_C_rho);

    grad = -ApplyMap(logm(PHI_rho), PHI_adj);
    grad = grad + ApplyMap(logm(PHI_C_rho), PHI_C_adj);
end


function A = rand_SPD(n)
    A = rand(n);
    A = A * A';
end


function H = relative_entropy(p, q)
    H = 0;
    for i = 1:length(p)
        if p ~= 0
            H = H + p(i) * log(p(i) / q(i));
        end
    end
end

function out = ApplyStinespring(rho, V, dim, sys, adj)
    % sys: 0

    if sys == 0
        if adj == 0
            out = PartialTrace(V * rho * V', 2, dim);
        else
            out = V' * kron(rho, eye(dim(2))) * V;
        end
    else
        if adj == 0
            out = PartialTrace(V * rho * V', 1, dim);
        else
            out = V' * kron(eye(dim(1)), rho) * V;
        end
    end
end

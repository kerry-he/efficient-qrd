clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))
addpath(genpath('YALMIP-master'))
addpath(genpath('ecos-matlab'))

n = 32; % Number of codewords 
N = 32; % Density matrix size

rng(7)
rho = zeros(N, N, n);
for i = 1:n
    rho(:, :, i) = RandomDensityMatrix(N, 1);
end

S_rho = zeros(n, 1);
for i = 1:n
    S_rho(i) = -von_neumann_entropy(rho(:, :, i));
end

Nc = 4; % Number of constraints
A = rand(Nc, n); 
b = rand(Nc, 1);

%% Algorithm
% Primal and dual variables
p = ones(n, 1) / n;
lambda = zeros(Nc, 1);
lambda_prev = lambda;

t_p = 10;
t_d = 10;

obj(1) = holevo_inf(p, rho, S_rho);

for k = 1:5000
    tic
    % Step sizes
    theta = 1.0;
    t_p = t_p*theta;
    t_d = t_d*theta;

    lambda_prev_prev = lambda_prev;

    p_prev = p;
    lambda_prev = lambda;

    Rho = zeros(N, N);
    for i = 1:n
        Rho = Rho + p(i)*rho(:, :, i);
    end
    sigma_log = logm(Rho);

    grad_p = zeros(n, 1);
    for i = 1:n
        grad_p(i) = quantum_relative_entropy_i(rho, sigma_log, S_rho, i) - 1;
    end    

    while true
        lambda_bar = lambda_prev + theta*(lambda_prev - lambda_prev_prev);

        for i = 1:n
            p(i) = p_prev(i) * exp(t_p*(grad_p(i) - lambda_bar'*A(:, i)));
        end
        p = p / sum(p);

        lambda = lambda_prev + t_d * (A*p - b); lambda = max(0, lambda);

        % Compute gradients
        obj(k+1) = holevo_inf(p, rho, S_rho);

        if -obj(k+1) <= -obj(k) - grad_p'*(p - p_prev) + (relative_entropy(p, p_prev) - sum(p) + sum(p_prev)) / t_p ...
                + sum((lambda-lambda_bar).^2, 'all')/(2*t_d) - ((lambda-lambda_bar)' * A * (p - p_prev))
            break
        end

        fprintf("BACKTRACKING\n")
        theta = theta * 0.9;
        t_p = t_p * 0.9;
        t_d = t_d * 0.9;
    end

    time(k) = toc;
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj(k))

    p_res(k) = norm(lambda - lambda_prev) / (t_d * max(1, max(abs(lambda))));
    d_res(k) = relative_entropy(p, p_prev) / (t_p);

%     if p_res(k) + d_res(k) <= 1e-7
%         break
%     end    

end

fprintf("Elapsed time is %.6f seconds.\n", sum(time))


%% Backtracking MD
yalmip('clear')
X = sdpvar(n, 1);
nu = sdpvar(Nc, 1);
lambda = sdpvar(1, 1);
% Constraints = [sum(X) == 1, A*X <= b];
Constraints = [nu >= 0];
options = sdpsettings('verbose', 0, 'solver', 'ecos');

p = ones(n, 1) / n;
t = 5;

obj_md(1) = holevo_inf(p, rho, S_rho);

for k = 1:5
    tic
    t_yalmip = 0;


    if k > 1
        grad_prev = grad_p;
    end
    p_prev = p;
    
    Rho = zeros(N, N);
    for i = 1:n
        Rho = Rho + p(i)*rho(:, :, i);
    end
    sigma_log = logm(Rho);

    grad_p = zeros(n, 1);
    for i = 1:n
        grad_p(i) = quantum_relative_entropy_i(rho, sigma_log, S_rho, i) - 1;
    end    

    while true
%         Objective = -t*grad_p' * X + kullbackleibler(X,p_prev);
        Objective = b'*nu + lambda + sum(p_prev.*exp(-A'*nu - lambda + t*grad_p));
        sol = optimize(Constraints,Objective,options);
%         p = value(X);
        p = p_prev.*exp(t*grad_p - A'*value(nu) - value(lambda));
        t_yalmip = t_yalmip + sol.yalmiptime;
        
        obj_md(k + 1) = holevo_inf(p, rho, S_rho);

        if t < 1 || -obj_md(k + 1) <= -obj_md(k) - grad_p'*(p - p_prev) + 1/t * relative_entropy(p, p_prev)
            break
        end

        t = t / 2;
    end


    Rho_new = zeros(N, N);
    for i = 1:n
        Rho_new = Rho_new + p(i)*rho(:, :, i);
    end

    if k > 2
        t = (relative_entropy(p, p_prev)) / quantum_relative_entropy(Rho, Rho_new);
        if isinf(t) || isnan(t) || t < 0
            t = 1;
        end
    end

    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_md(k+1))
    time_md(k) = toc - t_yalmip;
end


%% CVX

tic
cvx_begin
    cvx_precision(1e-4)
    variable x(n, 1)

    Rho = zeros(N, N);
    for i = 1:n
        Rho = Rho + x(i)*rho(:, :, i);
    end

    maximize ( quantum_entr( Rho ) + (x' * S_rho) );

    x >= 0;
    sum(x) == 1;
    A*x <= b;
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

function out = holevo_inf(p, rho, rho_S)
    [N, ~, n] = size(rho);

    Rho = zeros(N, N);
    for i = 1:n
        Rho = Rho + p(i)*rho(:, :, i);
    end

    out = von_neumann_entropy(Rho) + p' * rho_S;
end


function S = quantum_relative_entropy_i(rho, sigma_log, rho_S, i)
    S = rho_S(i) - trace(rho(:, :, i) * sigma_log);
end


function H = relative_entropy(p, q)
    H = 0;
    for i = 1:length(p)
        if p ~= 0
            H = H + p(i) * log(p(i) / q(i));
        end
    end
end
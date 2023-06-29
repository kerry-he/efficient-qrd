clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))
addpath(genpath('YALMIP-master'))
addpath(genpath('sdpt3-master'))
addpath(genpath('ecos-matlab'))

rng(5)

n = 5;
m = 5;
rho = RandomDensityMatrix(n*m, 1);
% rho = 0.5*eye(n*m)/(n*m) + 0.5*rho;
sigma = RandomDensityMatrix(n*m, 1);

t = 1 / max(eig(rho));

PPT = zeros((m*n)^2);
for i = 1:n
    for j = 1:n
        for k = 1:m
            for l = 1:m
                ij_mtx = zeros(n); ij_mtx(i, j) = 1;
                kl_mtx = zeros(m); kl_mtx(k, l) = 1;
        
                vec_in = kron(ij_mtx, kl_mtx);
                vec_out = kron(ij_mtx, kl_mtx');
        
                PPT = PPT + vec_out(:) * vec_in(:)';
            end
        end
    end
end
PPT = sparse(PPT);

% %% Constant step MD
% yalmip('clear')
% X = sdpvar(n*m, n*m, 'hermitian');
% Constraints = [X >= 0, trace(X) == 1, reshape(PPT * X(:), [n*m, n*m]) >= 0];
% options = sdpsettings('verbose',0);
% 
% for k = 1:2500
%     tic
%     [Ur, Dr] = eig(rho); log_rho = Ur * diag(log(diag(Dr))) * Ur';
%     [Us, Ds] = eig(sigma); log_sigma = Us * diag(log(diag(Ds))) * Us';
% 
%     obj(k) = trace(rho * (log_rho - log_sigma));
% 
%     L = lowener(Ds);
% 
%     grad_sigma = -Us * (L .* (Us'*rho*Us)) * Us';
% 
%     Objective = trace((t*grad_sigma + sigma^(-1)) * X) - logdet(X);
% %     Objective = trace(X * (grad_sigma - logm(sigma))) - entropy(eigv(X));
%     sol = optimize(Constraints,Objective,options);
%     sigma = value(X);
% 
%     fprintf("Iteration: %d \t Objective: %.5e\n", k, obj(k))
%     time(k) = toc - sol.yalmiptime;
% end



%% Backtracking MD
yalmip('clear')
X = sdpvar(n*m, n*m, 'hermitian');
g = sdpvar(n*m, n*m, 'hermitian');
Constraints = [X >= 0, trace(X) == 1, reshape(PPT * X(:), [n*m, n*m]) >= 0];
options = sdpsettings('verbose',0, 'solver', 'sdpt3');


[Ur, Dr] = eig(rho); log_rho = Ur * diag(log(diag(Dr))) * Ur';
[Us, Ds] = eig(sigma); log_sigma = Us * diag(log(diag(Ds))) * Us';

obj(1) = trace(rho * (log_rho - log_sigma));

for k = 1:100
    tic
    t_yalmip = 0;

    L = lowener(Ds);

    if k > 1
        grad_prev = grad_sigma;
    end
    sigma_prev = sigma;
    grad_sigma = -Us * (L .* (Us'*rho*Us)) * Us';

    while true

        Objective = trace((t*grad_sigma + sigma_prev^(-1)) * X) - logdet(X);
        sol = optimize(Constraints,Objective,options);
        sigma = value(X);
        t_yalmip = t_yalmip + sol.yalmiptime;

        [Ur, Dr] = eig(rho); log_rho = Ur * diag(log(diag(Dr))) * Ur';
        [Us, Ds] = eig(sigma); log_sigma = Us * diag(log(diag(Ds))) * Us';
        
        obj(k + 1) = trace(rho * (log_rho - log_sigma));

        if obj(k + 1) <= obj(k) + trace(grad_sigma * (sigma - sigma_prev)) + 1/t * logdetdist(sigma, sigma_prev)
            break
        end

        t = t / 2;
    end


    if k > 2
        t = (logdetdist(sigma, sigma_prev) + logdetdist(sigma_prev, sigma) / trace((grad_sigma - grad_prev) * (sigma - sigma_prev)));
        t = max(t, 1 / max(eig(rho)));
        if isinf(t) || isnan(t)
            t = 1 / max(eig(rho));
        end
    end

    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj(k))
    time(k) = toc - t_yalmip;
end



%% CVX

tic
cvx_begin sdp
    cvx_precision low
    variable X(n*m, n*m) symmetric
    minimize ( quantum_rel_entr( rho, X ) );

    X >= 0;
    trace(X) == 1;
    reshape(PPT * X(:), [n*m, n*m]) >= 0;
cvx_end
toc




%% Functions
function L = lowener(D)
    n = length(D);
    L = zeros(n, n);
    for i = 1:n
        for j = 1:n
            if D(i, i) == D(i, j)
%                 L(i, j) = 1 / D(i, i)^2;
                L(i, j) = 1 / D(i, i);
            else
%                 L(i, j) = 1 / (D(i, i) * D(j, j));
                L(i, j) = (log(D(i, i)) - log(D(j, j))) / (D(i, i) - D(j, j));
            end
        end
    end
end

function D = logdetdist(rho, sigma)
    rhosigma = rho * sigma^-1;
    D = trace(rhosigma) - log(det(rhosigma)) - length(rho);
end
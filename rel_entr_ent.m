clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))
addpath(genpath('YALMIP-master'))

rng(1)

n = 2;
m = 2;
rho = RandomDensityMatrix(n*m, 1);
% rho = 0.01*eye(n*m)/(n*m) + 0.99*rho;

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

%% Primal and dual variables
sigma = RandomDensityMatrix(n*m, 1);
nu = 1;
lambda = eye(n*m);
nu_prev = nu;
lambda_prev = lambda;

t_p = 2;
t_d = 2;

[Ur, Dr] = eig(rho); log_rho = Ur * diag(log(diag(Dr))) * Ur';
[Us, Ds] = eig(sigma); log_sigma = Us * diag(log(diag(Ds))) * Us';
obj(1) = trace(rho * (log_rho - log_sigma));

for k = 1:5000
    tic
    % Step sizes
    theta = 1.01;
    t_p = t_p*theta;
    t_d = t_d*theta;

    nu_prev_prev = nu_prev;
    lambda_prev_prev = lambda_prev;

    sigma_prev = sigma;
    nu_prev = nu;
    lambda_prev = lambda;
    log_sigma_prev = log_sigma;
    Ds_prev = Ds;
    Us_prev = Us;

    while true
        nu_bar = nu_prev + theta*(nu_prev - nu_prev_prev);
        lambda_bar = lambda_prev + theta*(lambda_prev - lambda_prev_prev);

        L = lowener(Ds_prev);
    
        grad_sigma = -Us_prev * (L .* (Us_prev'*rho*Us_prev)) * Us_prev';
        sigma = (sigma_prev^-1 + t_p * (grad_sigma + nu_bar*eye(n*m) - reshape(PPT * lambda_bar(:), [n*m, n*m])))^-1;

        grad_nu = trace(sigma) - 1;
        grad_lambda = -reshape(PPT * sigma(:), [n*m, n*m]);
        nu = nu_prev + t_d * grad_nu;
        lambda = lambda_prev + t_d * grad_lambda; [U, D] = eig(lambda); lambda = U * max(0, D) * U';

        % Compute gradients
        [Us, Ds] = eig(sigma); log_sigma = Us * diag(log(diag(Ds))) * Us';
    
        obj(k+1) = trace(rho * (log_rho - log_sigma));

        if prod(diag(Ds) > 0) && obj(k+1) <= obj(k) + trace(grad_sigma * (sigma - sigma_prev)) + logdetdist(sigma, sigma_prev) / t_p ...
                + (sum(lambda.^2, 'all') + nu^2)/(2*t_d) - ((nu-nu_prev) * trace(sigma - sigma_prev)) ...
                + trace((lambda - lambda_prev) * reshape(PPT * (sigma(:)-sigma_prev(:)), [n*m, n*m]))
            break
        end

        theta = theta * 0.75;
        t_p = t_p * 0.75;
        t_d = t_d * 0.75;
    end



    nu_hist(k) = nu;
    time(k) = toc;

    p_res(k) = ((sum((lambda - lambda_prev).^2, 'all') + (nu - nu_prev)^2)) /(2*t_d * max([1, nu, max(abs(lambda), [], 'all')]));
    d_res(k) = logdetdist(sigma, sigma_prev) / (t_p * max(1, max(sigma, [], 'all')));

    tr(k) = nu;

    if p_res(k) + d_res(k) <= 1e-7
        break
    end    
    
end

fprintf("Elapsed time is %.6f seconds.\n", sum(time))

%%

% clear all; close all; clc;
% 
% n = 9;
% rho = RandomDensityMatrix(n, 1);
% 
% t = 1;
% for i = [1 -1 1i -1i]
%     for j = [1 -1 1i -1i]
%         for k = [1 -1 1i -1i]
%             G{t} = kron(diag([i j k]), diag([i j k])') / 8;
%             t = t + 1;
%         end
%     end
% end
% 
% tes = zeros(n);
% for i = 1:64
%     tes = tes + G{i}' * G{i};
% end
% 
% 
% % G = KrausOperators(DepolarizingChannel(n, 0.25));
% % G = KrausOperators(RandomSuperoperator([n, n], 1, 1, 1));
% G_adj = G;
% for i = 1:length(G_adj)
%     G_adj{i} = G_adj{i}';
% end
% 
% % Primal and dual variables
% sigma = RandomDensityMatrix(n, 1);
% nu = 1;
% V = eye(n);
% 
% t_p = 0.1;
% t_d = 0.1;
% 
% tic
% for k = 1:50
%     [Ur, Dr] = eig(rho); log_rho = Ur * diag(log(diag(Dr))) * Ur';
%     [Us, Ds] = eig(sigma); log_sigma = Us * diag(log(diag(Ds))) * Us';
% 
%     obj(k) = trace(rho * (log_rho - log_sigma));
% 
%     L = lowener(Ds);
% 
%     grad_sigma = -Us * (L .* (Us'*rho*Us)) * Us' + nu*eye(n) + ApplyMap(V, G_adj) - V;
%     grad_nu = trace(sigma) - 1;
%     grad_V = ApplyMap(sigma, G) - sigma;
% 
%     sigma = (sigma^-1 + t_p * grad_sigma)^-1;
%     nu = nu + t_d * grad_nu;
%     V = V + t_d * grad_V;
% 
%     nu_hist(k) = nu;
% end
% toc


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

function D = qre(rho, sigma, log_rho, log_sigma)
    D = trace(rho * (log_rho - log_sigma)) - trace(rho - sigma);
end

function D = logdetdist(rho, sigma)
    rhosigma = rho * sigma^-1;
    D = trace(rhosigma) - log(det(rhosigma)) - length(rho);
end

function nu = tr_norm(lambda, nu)
    EPS = 1e-10;

    f = sum(1 ./ (lambda + nu));
    df = -sum(1 ./ (lambda + nu).^2);

    while abs(f) >= EPS
        nu = nu - f/df;
        f = sum(1 ./ (lambda + nu));
        df = -sum(1 ./ (lambda + nu).^2);
    end
end

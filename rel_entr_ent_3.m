clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))
addpath(genpath('YALMIP-master'))

rng(1)

n = 5;
m = 5;
rho = RandomDensityMatrix(n*m, 1);
rho = 0.1*eye(n*m)/(n*m) + 0.9*rho;

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
lambda_prev = lambda;

t_p = 10;
t_d = 10;

[Ur, Dr] = eig(rho); log_rho = Ur * diag(log(diag(Dr))) * Ur';
[Us, Ds] = eig(sigma); log_sigma = Us * diag(log(diag(Ds))) * Us';
obj(1) = trace(rho * (log_rho - log_sigma));

for k = 1:5000
    tic
    % Step sizes
    theta = 1.0;
    t_p = t_p*theta;
    t_d = t_d*theta;

    lambda_prev_prev = lambda_prev;

    sigma_prev = sigma;
    lambda_prev = lambda;
    log_sigma_prev = log_sigma;
    Ds_prev = Ds;
    Us_prev = Us;

    while true
        lambda_bar = lambda_prev + theta*(lambda_prev - lambda_prev_prev);

        L = lowener(Ds_prev);
    
        grad_sigma = -Us_prev * (L .* (Us_prev'*rho*Us_prev)) * Us_prev';

        X = sigma_prev^-1 + t_p * (grad_sigma - reshape(PPT * lambda_bar(:), [n*m, n*m]));
        [Us, Ds] = eig(X);
        nu = tr_norm(diag(Ds), t_p);
        Ds = diag(1./(diag(Ds) + nu));
        sigma = Us * Ds * Us';

        grad_lambda = -reshape(PPT * sigma(:), [n*m, n*m]);
        lambda = lambda_prev + t_d * grad_lambda; [U, D] = eig(lambda); lambda = U * max(0, D) * U';

        % Compute gradients
        log_sigma = Us * diag(log(diag(Ds))) * Us';
    
        obj(k+1) = trace(rho * (log_rho - log_sigma));

        if prod(diag(Ds) > 0) && obj(k+1) <= obj(k) + trace(grad_sigma * (sigma - sigma_prev)) + logdetdist(sigma, sigma_prev) / t_p ...
                + (sum((lambda - lambda_prev).^2, 'all'))/(2*t_d) ...
                + trace((lambda - lambda_prev) * reshape(PPT * (sigma(:)-sigma_prev(:)), [n*m, n*m]))
            break
        end

        theta = theta * 0.9;
        t_p = t_p * 0.9;
        t_d = t_d * 0.9;
    end


    time(k) = toc;

    p_res(k) = ((sum((lambda - lambda_prev).^2, 'all'))) /(2*t_d * max([1, max(abs(lambda), [], 'all')]));
    d_res(k) = logdetdist(sigma, sigma_prev) / (t_p * max(1, max(sigma, [], 'all')));

%     if p_res(k) + d_res(k) <= 1e-7
%         break
%     end    
%     
end

fprintf("Elapsed time is %.6f seconds.\n", sum(time))

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

    f = sum(1 ./ (lambda + nu)) - 1;
    df = -sum(1 ./ (lambda + nu).^2);

    while abs(f) >= EPS
        nu = nu - f/df;
        f = sum(1 ./ (lambda + nu)) - 1;
        df = -sum(1 ./ (lambda + nu).^2);
    end
end

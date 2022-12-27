clear all; close all; clc;

m = 16;      % Number of inputs
n = 16;      % Number of outputs

rng(3)
Q = rand(n, m);
Q = Q ./ sum(Q);

% p = ones(m, 1) / m;
% p = zeros(m, 1) / m; p(1:2) = [0.9 0.1];
p = rand(m, 1); p = p / sum(p);

rho = rand(n, m);
% U = RandomUnitary(n, 1);
% d = rand(m, 1); d = d / sum(d);
% rho = U * diag(d) * U';
% rho = ones(n) - eye(n);

lambda = 1.0;

%% Optimization algorithm
M = 2000;
obj = zeros(M, 1);
lagrangian = zeros(M, 1);
lambda_hist = zeros(M, 1);
lambda_2_hist = zeros(M, 1);

Q_std = Q;
for k = 1:M
    % Perform Blahut-Arimoto iteration
    Q_prev = Q_std;
    q = Q_std*p;

    for i = 1:n
        for j = 1:m
            Q_std(i, j) = q(i) * exp(-lambda * rho(i, j));
        end
    end
    Q_std = Q_std ./ sum(Q_std);

    constraint = 0;
    for i = 1:n
        for j = 1:m
            constraint = constraint + p(j) * Q_std(i, j) * rho(i, j);
        end
    end
%     lambda = max(lambda + (constraint - D), 0);
%     lambda_hist(k) = lambda;

    % Compute lower bound
    q = Q_std*p;

    for i = 1:n
        for j = 1:m
            grad(i, j) = p(j) * (log(Q_std(i, j) / q(i)) + lambda*rho(i, j));
        end
    end
    lb(k) = sum(min(grad));

    % Compute objective value
    obj(k) = mutual_information(p, Q_std);
    lagrangian(k) = obj(k) + lambda * constraint;
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj(k))
end

% Q_test = Q;
% lambda_test = 3.0;
% obj_test = zeros(M, 1);
% 
% A = exp(-lambda * rho);
% 
% q = Q * p;
% 
% for k = 1:M
%     Tu(k) = 0;
%     q_prev = q;
%     for i = 1:n
%         c(i) = 0;
%         for j = 1:m
%             c(i) = c(i) + p(j) * A(i, j) / (q_prev' * A(:, j));
%         end
% 
%         q(i) = q_prev(i) * c(i);
%     end
% 
%     for i = 1:n
%         Tu(k) = Tu(k) - q(i) * log(c(i));
%     end
% 
%     Tl(k) = -max(log(c));
% 
%     gap(k) = Tu(k) - Tl(k);
%     fprintf("Iteration: %d \t Objective: %.5e\n", k, gap(k))
% end



% semilogy(abs(obj - min(obj(end), obj_test(end))))
% hold on
% semilogy(abs(obj_test - min(obj(end), obj_test(end))))
semilogy(obj - obj(end))
hold on
semilogy(lagrangian - lagrangian(end))
semilogy(lagrangian - lb')


%% Functions
function H = relative_entropy(p, q)
    H = 0;
    for i = 1:length(p)
        if p(i) ~= 0
            H = H + p(i) * log(p(i) / q(i));
        end
    end
end


function I = mutual_information(p, Q)
    q = Q*p;

    I = 0;
    for i = 1:length(p)
        I = I + p(i) * relative_entropy(Q(:, i), q);
    end
end
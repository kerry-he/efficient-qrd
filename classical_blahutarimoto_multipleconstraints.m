clear all; close all; clc;

m = 64;      % Number of inputs
n = 64;      % Number of outputs

rng(1)
Q = rand(n, m);
Q = Q ./ sum(Q);

P = rand(n, m);
P = P ./ sum(P);

p = ones(m, 1) / m;
p = rand(m, 1); p = p / sum(p);

% N = 1; % Number of constraints
% s = rand(N, m);
% C = [0.3];
N = 16; % Number of constraints
s = rand(N, m); 
C = rand(N, 1)*2.0;

L = max(vecnorm(s, 2, 1));

%% Optimization algorithm
M = 10000;
gamma = 1;
obj_std = zeros(M, 1);
lagrangian = zeros(M, 1);
lambda_hist = zeros(M, 1);


p_std = p;
lambda_std = zeros(N, 1);
tic
for k = 1:M
    % Perform Blahut-Arimoto iteration
    q = Q*p_std;
    p_prev = p_std;
    for i = 1:m
        p_std(i) = p_std(i) * exp((relative_entropy(Q(:, i), q) - lambda_std'*s(:, i)));
    end
   
    p_std = p_std / sum(p_std); % Normalise probability vector

    for i = 1:N
        lambda_std(i) = max(lambda_std(i) + (s(i, :)*p_std - C(i)), 0);
        lambda_hist(k, i) = lambda_std(i);
    end

    % Compute objective value
    obj_std(k) = mutual_information(p_std, Q);
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_std(k))
end
toc
% Alternative method
p_alt = p;
lambda_alt = ones(N, 1);

tau = 1 / L;
sigma = 1 / L;

obj_alt = zeros(M, 1);
lambda_alt_hist = zeros(M, 1);
tic
for k = 1:M
    p_prev = p_alt;

    % Perform Blahut-Arimoto iteration
    q = Q*p_alt;
    p_prev = p_alt;
    for i = 1:m
        p_alt(i) = p_alt(i) * exp((relative_entropy(Q(:, i), q) - lambda_alt'*s(:, i)) * tau);
    end
   
    p_alt = p_alt / sum(p_alt); % Normalise probability vector

    for i = 1:N
        lambda_alt(i) = max(lambda_alt(i) + (s(i, :)*(2*p_alt-p_prev) - C(i)) * sigma, 0);
        lambda_hist(k, i) = lambda_alt(i);
    end

    % Compute objective value
    obj_alt(k) = mutual_information(p_alt, Q);
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_alt(k))
end
% for k = 1:M
%     % Perform Blahut-Arimoto iteration
%     q = Q*p_alt;
%     p_prev = p_alt;
%     for i = 1:m
%         p_alt(i) = p_alt(i) * exp(relative_entropy(Q(:, i), q) / gamma);
%     end
% 
%     % Root-find for lambda
%     eps = 1e-10;
%     F = zeros(N, 1);
%     for i = 1:N
%         F(i) = sum((C(i) - s(i, :)').*p_alt.*exp(-lambda_alt'*s / gamma)');
%     end
%     while norm(F) > eps
%         dF = zeros(N, N);
%         for i = 1:N
%             for j = 1:N
%                 dF(i, j) = sum(-s(j, :)'.*(C(i) - s(i, :)').*p_alt.*exp(-lambda_alt'*s / gamma)') / gamma;
%             end
%         end
%         
%         lambda_alt = lambda_alt - dF\F;
% 
%         for i = 1:N
%             F(i) = sum((C(i) - s(i, :)').*p_alt.*exp(-lambda_alt'*s / gamma)');
%         end
%     end
% 
%     for i = 1:m
%         p_alt(i) = p_alt(i) .* exp(-lambda_alt'*s(:, i) / gamma);
%     end
%     p_alt = p_alt / sum(p_alt); % Normalise probability vector
% 
%     for i = 1:N
%         lambda_alt_hist(k, i) = lambda_alt(i);
%     end
% 
%     % Compute objective value
%     obj_alt(k) = mutual_information(p_alt, Q);
%     fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_alt(k))
% end
toc



% plot(obj_logger)
semilogy(abs(max(obj_std(end), obj_alt(end)) - obj_std))
hold on
semilogy(abs(max(obj_std(end), obj_alt(end)) - obj_alt))


%% Functions
function H = relative_entropy(p, q)
    H = 0;
    for i = 1:length(p)
        H = H + p(i) * log(p(i) / q(i));
    end
end

function I = mutual_information(p, Q)
    q = Q*p;

    I = 0;
    for i = 1:length(p)
        I = I + p(i) * relative_entropy(Q(:, i), q);
    end
end
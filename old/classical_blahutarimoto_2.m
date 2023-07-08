clear all; close all; clc;
addpath(genpath('YALMIP-master'))
addpath(genpath('ecos-matlab'))

m = 1024;      % Number of inputs
n = 1024;      % Number of outputs

rng(2)
Q = rand(n, m);
Q = Q ./ sum(Q);

grad_p_freeze = sum(Q.*log(Q))';


N = 8; % Number of constraints
A = rand(N, m); 
b = rand(N, 1);


%% Optimization algorithm

% Primal and dual variables
p = ones(m, 1) / m;
lambda = zeros(N, 1);
lambda_prev = lambda;

t_p = 10;
t_d = 10;

obj(1) = mutual_information(p, Q);

for k = 1:1000
    tic
    % Step sizes
    theta = 1.01;
    t_p = t_p*theta;
    t_d = t_d*theta;

    lambda_prev_prev = lambda_prev;

    p_prev = p;
    lambda_prev = lambda;

    grad_p = grad_p_freeze - sum(Q.*log(Q*p_prev))';

    while true
        lambda_bar = lambda_prev + theta*(lambda_prev - lambda_prev_prev);

        for i = 1:m
            p(i) = p_prev(i) * exp(t_p*(grad_p(i) - lambda_bar'*A(:, i)));
        end
        p = p / sum(p);

        lambda = lambda_prev + t_d * (A*p - b); lambda = max(0, lambda);

        % Compute gradients
        obj(k+1) = mutual_information(p, Q);

        if -obj(k+1) <= -obj(k) - grad_p'*(p - p_prev) + (relative_entropy(p, p_prev) - sum(p) + sum(p_prev)) / t_p ...
                + sum((lambda-lambda_bar).^2, 'all')/(2*t_d) - ((lambda-lambda_bar)' * A * (p - p_prev))
            break
        end

        display("BACKTRACKING")

        theta = theta * 0.75;
        t_p = t_p*theta;
        t_d = t_d*theta;
    end

    time(k) = toc;
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj(k))

    p_res(k) = norm(lambda - lambda_prev)^2 / (2*t_d * max(1, max(lambda)));
    d_res(k) = relative_entropy(p, p_prev) / t_p;
%     d_res(k) = norm(log(p) - log(p_prev)) / t_p;

    if p_res(k) + d_res(k) <= 1e-7
        break
    end
    

end


%%
% Primal and dual variables
% p = ones(m, 1) / m;
% lambda = zeros(N, 1);
% lambda_prev = lambda;
% 
% t_p = 10;
% t_d = 25;
% 
% obj_alt(1) = mutual_information(p, Q);
% 
% tic
% for k = 1:100
%     % Step sizes
%     theta = 1.00;
%     t_p = t_p*theta;
%     t_d = t_d*theta;
% 
%     lambda_prev_prev = lambda_prev;
% 
%     p_prev = p;
%     lambda_prev = lambda;
% 
%     while true
%         lambda_bar = lambda_prev + theta*(lambda_prev - lambda_prev_prev);
%         
%         grad_p = zeros(m, 1);
%         for i = 1:m
%             grad_p(i) = relative_entropy(Q(:, i), Q*p_prev) - 1;
%         end
% 
%         for i = 1:m
%             p(i) = p_prev(i) * exp(t_p*(relative_entropy(Q(:, i), Q*p_prev) - lambda_bar'*A(:, i)));
%         end
%         p = p / sum(p);
% 
%         lambda = lambda_prev + t_d * (A*p - b); lambda = max(0, lambda);
% 
%         % Compute gradients
%         obj_alt(k+1) = mutual_information(p, Q);
% 
%         if 0 <= (relative_entropy(p, p_prev) - sum(p) + sum(p_prev)) * ( 1/t_p - 1 ) ...
%                 + sum((lambda-lambda_bar).^2, 'all')/(2*t_d) - ((lambda-lambda_bar)' * A * (p - p_prev))
%             break
%         end
% 
%         theta = theta * 0.5;
%         t_p = t_p*theta;
%         t_d = t_d*theta;
%     end
%     
% end
% toc


%% Backtracking MD
yalmip('clear')
g = sdpvar(m, 1);
X = sdpvar(m, 1);
nu = sdpvar(N, 1);
lambda = sdpvar(1, 1);
% Constraints = [sum(X) == 1, A*X <= b];
Constraints = [nu >= 0];
options = sdpsettings('verbose', 0, 'solver', 'ecos');

% g = sdpvar(m, 1);
% x = sdpvar(m, 1);
% 
% Constraints = [nu >= 0, -A'*nu - lambda == x];
% Objective = b'*nu + lambda + g'*exp(x);
% P=optimizer(Constraints, Objective, sdpsettings('solver', 'fmincon'), g, {nu, lambda});


p = ones(m, 1) / m;
t = 5;

obj_md(1) = mutual_information(p, Q);

for k = 1:5
    tic
    t_yalmip = 0;


    if k > 1
        grad_prev = grad_p;
    end
    p_prev = p;
    grad_p = zeros(m, 1);
    for i = 1:m
        grad_p(i) = relative_entropy(Q(:, i), Q*p_prev) - 1;
    end

    while true
%         Objective = -t*grad_p' * X + kullbackleibler(X,p_prev);
%         p = value(X);
        Objective = b'*nu + lambda + sum(p_prev.*exp(-A'*nu - lambda + t*grad_p));
        sol = optimize(Constraints,Objective,options);
        p = p_prev.*exp(t*grad_p - A'*value(nu) - value(lambda));
        t_yalmip = t_yalmip + sol.yalmiptime;

%         sol = P(exp(t*grad_p) .* p_prev);
%         nu_sol = sol{1}; lambda_sol = sol{2};        
%         p = p_prev.*exp(t*grad_p - A'*value(nu_sol) - value(lambda_sol));

        
        obj_md(k + 1) = mutual_information(p, Q);

        if t < 1 || -obj_md(k + 1) <= -obj_md(k) - grad_p'*(p - p_prev) + 1/t * relative_entropy(p, p_prev)
            break
        end

        t = t / 2;
    end


    if k > 2
        t = (relative_entropy(p, p_prev)) / (relative_entropy(Q*p, Q*p_prev));
        if isinf(t) || isnan(t) || t < 0
            t = 1;
        end
    end

    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_md(k+1))
    time_md(k) = toc - t_yalmip;
end

plot(obj)
hold on
% plot(obj_alt)
plot(obj_md)

figure

semilogy(abs(obj-obj(end)))
hold on
% semilogy(abs(obj_alt-obj(end)))
semilogy(abs(obj_md-obj(end)))

%%
tic
yalmip('clear')
x = sdpvar(m, 1);
Constraints = [sum(x) == 1, A*x <= b, x >= 0];
options = sdpsettings('verbose', 1, 'solver', 'ecos');

Objective = -sum(entropy(Q*x)) + sum(x .* sum(-Q.*log(Q))');
sol = optimize(Constraints,Objective,options);
toc

figure
semilogy(cumsum(time), abs(obj(2:end) + value(Objective)))
hold on
semilogy(cumsum(time_md), abs(obj_md(2:end) + value(Objective)))


%% Functions
function H = relative_entropy(p, q)
%     H = 0;
%     for i = 1:length(p)
%         H = H + p(i) * log(p(i) / q(i));
%     end
    H = sum(p .* log(p ./ q));
end

function I = mutual_information(p, Q)
    q = Q*p;

    I = 0;
    for i = 1:length(p)
        I = I + p(i) * relative_entropy(Q(:, i), q);
    end
end
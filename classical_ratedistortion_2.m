clear all; close all; clc;

m = 32;      % Number of inputs
n = 32;      % Number of outputs

rng(3)

% p = ones(m, 1) / m;
% p = zeros(m, 1) / m; p(1:2) = [0.9 0.1];
p = rand(m, 1); p = p / sum(p);

P = rand(n, m);
P = P ./ sum(P) .* p';

% rho = rand(n, m);
% U = RandomUnitary(n, 1);
% d = rand(m, 1); d = d / sum(d);
% rho = U * diag(d) * U';
delta = ones(n) - eye(n);

% p = [0.5; 0.3; 0.2];
% delta = [0 1 2; 1 2 0; 3 0 1]';


%% Optimization algorithm
D = 0.5;
lambda = 5;
lambda_prev = lambda;

t_p = 3;
t_d = 30;

obj(1) = mutual_information(p, P./p');


for k = 1:1000
    tic
    % Step sizes
    theta = 1.01;
    t_p = t_p*theta;
    t_d = t_d*theta;

    lambda_prev_prev = lambda_prev;

    P_prev = P;
    lambda_prev = lambda;

    while true
        lambda_bar = lambda_prev + theta*(lambda_prev - lambda_prev_prev);
        
        grad_P = log(P_prev) - log(p' .* sum(P_prev, 2));

        for i = 1:n
            for j = 1:m
                P(i, j) = P_prev(i, j) * exp(-(grad_P(i, j) + lambda_bar*delta(i, j)) * t_p);
            end
        end
        P = P ./ sum(P) .* p';

        lambda = lambda_prev + t_d * (sum(delta .* P, 'all') - D); lambda = max(0, lambda);

        % Compute gradients
        obj(k+1) = mutual_information(p, P./p');

        if obj(k+1) <= obj(k) + sum(grad_P.*(P - P_prev), 'all') + abs(relative_entropy(P(:), P_prev(:))) / t_p ...
                + (lambda-lambda_bar)^2/(2*t_d) - (lambda-lambda_bar) * sum(delta .* (P - P_prev), 'all')
            break
        end

        theta = theta * 0.75;
        t_p = t_p*theta;
        t_d = t_d*theta;
    end

    time(k) = toc;
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj(k))    
    
    p_res(k) = norm(lambda - lambda_prev) / (t_d * max(1, max(abs(lambda))));
    d_res(k) = relative_entropy(P, P_prev) / (t_p);

    if p_res(k) + d_res(k) <= 1e-7
        break
    end    

end



%%
yalmip('clear')
X = sdpvar(n, m, 'full');
Constraints = [sum(X)' == p, X >= 0, sum(X .* delta, 'all') <= D];
options = sdpsettings('verbose', 1, 'solver', 'ecos', 'ecos.maxit', 200);

Objective = kullbackleibler( X, sum(X, 2) * p' );
sol = optimize(Constraints,Objective,options);


semilogy(cumsum(time), abs(obj(2:end) - value(Objective)))

%% Functions
function H = relative_entropy(p, q)
    H = 0;
    for i = 1:length(p)
        if p(i) ~= 0
            H = H + p(i) * log(p(i) / q(i));
        end
    end
%     H = sum(p .* log( p ./ q ), 'all');
end


function I = mutual_information(p, Q)
    q = Q*p;

    I = 0;
    for i = 1:length(p)
        I = I + p(i) * relative_entropy(Q(:, i), q);
    end
end
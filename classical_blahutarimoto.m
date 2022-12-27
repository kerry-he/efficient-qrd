clear all; close all; clc;

m = 128;      % Number of inputs
n = 128;      % Number of outputs

rng(543)
Q = rand(n, m);
Q = Q ./ sum(Q);

P = rand(n, m);
P = P ./ sum(P);

p = ones(m, 1) / m;
p = rand(m, 1); p = p / sum(p);

s = rand(m, 1)*0;
S = 0.3;


%% Optimization algorithm
M = 500;
gamma = 1.0;
obj_std = zeros(M, 1);
lagrangian = zeros(M, 1);
lambda_hist = zeros(M, 1);

p_std = p;
lambda_std = 0;
tic
for k = 1:M
    % Perform Blahut-Arimoto iteration
    q = Q*p_std;
    p_prev = p_std;
    for i = 1:m
        grad(i) = relative_entropy(Q(:, i), q);
        p_std(i) = p_std(i) * exp((grad(i) - lambda_std*s(i)) / gamma);
    end
   
    p_std = p_std / sum(p_std); % Normalise probability vector

    lambda_std = max(lambda_std + (s'*p_std - S) / gamma, 0);
    lambda_hist(k) = lambda_std;

    % Compute objective value
    obj_std(k) = mutual_information(p_std, Q);
    ub_std(k) = max(grad);
    lagrangian(k) = obj_std(k) + lambda_std * (s'*p_std - S);
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_std(k))
end
toc

% Heuristic
p_heur = p;
L_heur = 1.0;
tic
for k = 1:M
    % Perform Blahut-Arimoto iteration
    q = Q*p_heur;
    p_prev = p_heur;
    for i = 1:m
        grad(i) = relative_entropy(Q(:, i), q);
    end
    for i = 1:m
        p_heur(i) = p_heur(i) * exp((grad(i) - max(grad)) / L_heur);
    end

    p_heur = p_heur / sum(p_heur); % Normalise probability vector

    % Estimate L
    L_heur = relative_entropy(Q*p_heur, q) / relative_entropy(p_heur, p_prev);
    if L_heur < 0 || L_heur > 1 || isnan(L_heur)
        L_heur = 1.0;
    end    

    % Compute objective value
    obj_heur(k) = mutual_information(p_heur, Q);
    ub_heur(k) = max(grad);
    fprintf("Iteration: %d \t Objective: %.5e \t L: %.5f\n", k, obj_heur(k), L_heur)
end
toc


% Backtrack line search
p_ls = p;
L_ls = 1.0;
obj_0 = mutual_information(p_ls, Q);
tic
for k = 1:M
    % Perform Blahut-Arimoto iteration
    q = Q*p_ls;
    p_prev = p_ls;
    for i = 1:m
        grad(i) = relative_entropy(Q(:, i), q);
    end

    for t = 1:1000
        for i = 1:m
            p_ls(i) = p_prev(i) * exp((grad(i) - max(grad)) / L_ls);
        end
        p_ls = p_ls / sum(p_ls); % Normalise probability vector

        obj_ls(k) = mutual_information(p_ls, Q);

        if k == 1
            obj_prev = obj_0;
        else
            obj_prev = obj_ls(k - 1);
        end
        if -obj_ls(k) <= -obj_prev - (grad * (p_ls - p_prev)) + L_ls*relative_entropy(p_ls, p_prev)
            break
        end
        L_ls = L_ls * 2.0;
        if L_ls > 1
            break
        end          
    end


    % Estimate L
    L_ls = relative_entropy(Q*p_ls, q) / relative_entropy(p_ls, p_prev);
    if L_ls < 0 || L_ls > 1 || isnan(L_ls)
        L_ls = 1.0;
    end    

    % Compute objective value
    ub_ls(k) = max(grad);
    fprintf("Iteration: %d \t Objective: %.5e \t L: %.5f\n", k, obj_ls(k), L_ls)
end
toc


% Accelerated method
p_acc = p;
L_acc = 1.0;
obj_0 = mutual_information(p_acc, Q);
x = p;
z = p;
theta = 1;
L_acc = 0.001;
tic
for k = 1:M
    % Perform Blahut-Arimoto iteration
    y = (1 - theta) * x + theta * z;

    q = Q*y;
    for i = 1:m
        grad(i) = relative_entropy(Q(:, i), q);
    end

    x_prev = x;
    z_prev = z;
    % Line search
    for t = 1:1000
        for i = 1:m
            z(i) = z_prev(i) * exp((grad(i) - max(grad)) / L_acc);
        end
        z = z / sum(z); % Normalise probability vector
        x = (1 - theta) * x_prev + theta * z;        

        obj_acc(k) = mutual_information(x, Q);

        if k == 1
            obj_prev = obj_0;
        else
            obj_prev = obj_acc(k - 1);
        end
        if -obj_acc(k) <= -obj_prev - (grad * (x - x_prev)) + L_acc*relative_entropy(x, x_prev)
            break
        end
        L_acc = L_acc * 1.1;
        
        if L_acc > 1
            L_acc = 1.0;
            break
        end     
    end

    a = 1 / (theta^2);
    b = 1;
    c = -1;
    theta = (-b + sqrt(b^2 - 4*a*c)) / (2*a);    

    obj_acc(k) = mutual_information(x, Q);
    obj_acc_z(k) = mutual_information(z, Q);
    
    if k >= 10
        if obj_acc(k) < obj_acc(k-1)
            theta = 1;
            z = x;
            L_acc = 0.001;
        end
    end

    % Compute objective value
    ub_acc(k) = max(grad);
    fprintf("Iteration: %d \t Objective: %.5e \t L: %.5f \t theta: %.5f\n", k, obj_acc(k), L_acc, theta)
end
toc




% Root finding constraint method
p_alt = p;
lambda_alt = 0;

obj_alt = zeros(M, 1);
lambda_alt_hist = zeros(M, 1);
tic
for k = 1:M
    % Perform Blahut-Arimoto iteration
    q = Q*p_alt;
    p_prev = p_alt;
    for i = 1:m
        p_alt(i) = p_alt(i) * exp(relative_entropy(Q(:, i), q) / gamma);
    end

    % Root-find for lambda
    eps = 1e-10;
    f = sum((S - s).*p_alt.*exp(-lambda_alt * s / gamma));
    while abs(f) > eps
        df = sum(-s.*(S - s).*p_alt.*exp(-lambda_alt * s / gamma)) / gamma;
        if df < 0
            lambda_alt = 0;
            f = sum((S - s).*p_alt.*exp(-lambda_alt * s / gamma));
            break
        end
        lambda_alt = lambda_alt - f / df;
%         if lambda_alt < 0
%             lambda_alt = 0;
%             break
%         end
        f = sum((S - s).*p_alt.*exp(-lambda_alt * s / gamma));
    end
   
    p_alt = p_alt .* exp(-lambda_alt * s / gamma);
    p_alt = p_alt / sum(p_alt); % Normalise probability vector

    lambda_alt = max(lambda_alt + (s'*p_alt - S), 0);
    lambda_alt_hist(k) = lambda_alt;
    constr_hist(k) = f;

    % Compute objective value
    obj_alt(k) = mutual_information(p_alt, Q);
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_alt(k))
end
toc


semilogy(min([ub_std, ub_heur, ub_ls, ub_acc]) - obj_std)
hold on
semilogy(min([ub_std, ub_heur, ub_ls, ub_acc]) - obj_heur)
semilogy(min([ub_std, ub_heur, ub_ls, ub_acc]) - obj_ls)
semilogy(min([ub_std, ub_heur, ub_ls, ub_acc]) - obj_acc)

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
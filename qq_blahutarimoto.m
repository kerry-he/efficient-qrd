clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))

rng(2)
N = 32; % Density matrix size
gamma = 0.3;

rho = ones(N, N) / N;
rho = rand_SPD(N); rho = rho / trace(rho);

PHI = KrausOperators(RandomSuperoperator([N, N], 1, 1, 1));
PHI_C = ComplementaryMap(PHI, [N, N]);

PHI_adj = PHI;
for i = 1:length(PHI_adj)
    PHI_adj{i} = PHI_adj{i}';
end
PHI_C_adj = PHI_C;
for i = 1:length(PHI_C_adj)
    PHI_C_adj{i} = PHI_C_adj{i}';
end

display("DONE")

%% Setup algorithm
K = 100;
L = 2;

obj_heur = zeros(K, 1);
ub_heur = ones(K, 1)*Inf;
obj_std = zeros(K, 1);
ub_std = ones(K, 1)*Inf;
obj_ls = zeros(K, 1);
ub_ls = ones(K, 1)*Inf;
obj_acc = zeros(K, 1);
ub_acc = ones(K, 1)*Inf;


% Heuristic step size
L_heur = L;
rho_heur = rho;
[obj_heur(1), grad] = q_mutual_inf(rho_heur, PHI, PHI_adj, PHI_C, PHI_C_adj);
for k = 2:K
    rho_prev = rho_heur;
    
    % Perform Blahut-Arimoto iteration
    rho_heur = expm(logm(rho_heur) + grad / L_heur);
    rho_heur = rho_heur / trace(rho_heur);
    
    % Compute gradiate and function value
    grad_prev = grad;
    [obj_heur(k), grad] = q_mutual_inf(rho_heur, PHI, PHI_adj, PHI_C, PHI_C_adj);

    % Adaptive step size heuristic
    L_heur = trace(rho_heur * (grad_prev - grad)) / quantum_relative_entropy(rho_heur, rho_prev);
    if L_heur < 0 || L_heur > 2
        L_heur = 2.0;
    end    
    
    % Compute objective value
    fprintf("Iteration: %d \t Objective: %.5e \t L: %.4f\n", k, obj_heur(k), L_heur)
    
    ub_heur(k) = real(max(eig(grad)));
end

% Line search
L_ls = 0.2;
rho_ls = rho;
% for k = 1:K
%     [obj_0, grad] = q_mutual_inf(rho_ls, PHI, PHI_adj, PHI_C, PHI_C_adj);
% 
%     % Perform Blahut-Arimoto iteration
% %     rho = expm(logm(rho) + 1/L * grad);
% %     rho = rho / trace(rho);    
%     % Line search
%     rho_prev = rho_ls;
%     for t = 0:10000
%         rho_ls = expm(logm(rho_prev) + grad / L_ls);
%         rho_ls = rho_ls / trace(rho_ls);   
% 
%         [obj_ls(k), ~] = q_mutual_inf(rho_ls, PHI, PHI_adj, PHI_C, PHI_C_adj);
%         if k == 1
%             obj_prev = obj_0;
%         else
%             obj_prev = obj_ls(k - 1);
%         end
%         if -obj_ls(k) <= -obj_prev - trace(grad * (rho_ls - rho_prev)) + L_ls*quantum_relative_entropy(rho_ls, rho_prev)
%             break
%         end
%         L_ls = L_ls * 1.1;
%         if L_ls > 1
%             L_ls = 1.0;
%             break
%         end        
%     end    
% 
%     % Compute objective value
%     fprintf("Iteration: %d \t Objective: %.5e \t L: %.4f\n", k, obj_ls(k), L_ls)
%     
%     ub_ls(k) = real(max(eig(grad)));
% end
[obj_ls(1), grad] = q_mutual_inf(rho_ls, PHI, PHI_adj, PHI_C, PHI_C_adj);
for k = 2:K
    rho_prev = rho_ls;
    
    % Perform Blahut-Arimoto iteration
%     rho_ls = expm(logm(rho_ls) + grad / L_ls);
%     rho_ls = rho_ls / trace(rho_ls);
    for t = 0:10000
        rho_ls = expm(logm(rho_prev) + grad / L_ls);
        rho_ls = rho_ls / trace(rho_ls);   

        [obj_ls(k), ~] = q_mutual_inf(rho_ls, PHI, PHI_adj, PHI_C, PHI_C_adj);
        if k == 1
            obj_prev = obj_0;
        else
            obj_prev = obj_ls(k - 1);
        end
        if real(-obj_ls(k)) <= real(-obj_prev - trace(grad * (rho_ls - rho_prev)) + L_ls*quantum_relative_entropy(rho_ls, rho_prev))
            break
        end
        L_ls = L_ls * 1.5;
        if L_ls > 2
            break
        end        
    end        
    
    % Compute gradiate and function value
    grad_prev = grad;
    [obj_ls(k), grad] = q_mutual_inf(rho_ls, PHI, PHI_adj, PHI_C, PHI_C_adj);

    % Adaptive step size heuristic
    L_ls = trace(rho_ls * (grad_prev - grad)) / quantum_relative_entropy(rho_ls, rho_prev);
    if L_ls < 0 || L_ls > 2
        L_ls = 2.0;
    end
    
    % Compute objective value
    fprintf("Iteration: %d \t Objective: %.5e \t L: %.4f\n", k, obj_ls(k), L_ls)
    
    ub_ls(k) = real(max(eig(grad)));
end

% Accelerated method search
L_acc = 0.2;
rho_acc = rho;
theta = 1;
x = rho;
z = rho;
for k = 1:K
    y = (1 - theta) * x + theta * z;
    [obj_0, grad] = q_mutual_inf(y, PHI, PHI_adj, PHI_C, PHI_C_adj);

    % Perform Blahut-Arimoto iteration
    x_prev = x;
    z_prev = z;
    for t = 0:10000
        z = expm(logm(z_prev) + grad / L_acc);
        z = z / trace(z);
        x = (1 - theta) * x_prev + theta * z;
        
        [obj_acc(k), ~] = q_mutual_inf(x, PHI, PHI_adj, PHI_C, PHI_C_adj);
        if k == 1
            obj_prev = obj_0;
        else
            obj_prev = obj_acc(k - 1);
        end
        if real(-obj_acc(k)) <= real(-obj_prev - trace(grad * (x - x_prev)) + L_acc*quantum_relative_entropy(x, x_prev))
            break
        end
        L_acc = L_acc * 1.1;
        if L_acc > 2
            L_acc = 2.0;
            break
        end        
    end    
    
    a = 1 / (theta^2);
    b = 1;
    c = -1;
    theta = (-b + sqrt(b^2 - 4*a*c)) / (2*a);
    
    obj_acc_z(k) = q_mutual_inf(z, PHI, PHI_adj, PHI_C, PHI_C_adj);
    
    if k >= 3
    if obj_acc(k) < obj_acc(k-1)
        theta = 1;
        z = x;
    end
    end    

    % Compute objective value
    fprintf("Iteration: %d \t Objective: %.5e \t L: %.4f \t theta: %.4f\n", k, obj_acc(k), L_acc, theta)
    
    ub_acc(k) = real(max(eig(grad)));
end
% A = 1 / L;
% a = 1 / L;
% mu = 0.25;
% epsilon = 1;
% L_acc = 0.2;
% rho_acc = rho;
% theta = 1;
% x = rho;
% z = rho;
% for k = 1:K
%     A_prev = A;
%     a = (1 + A*mu)/(2*L) + sqrt((1+A*mu)/(4*L^2) + (A*(1+A*mu))/L);
%     A = A + a;
%     
%     y = A_prev/A * x + a/A * z;
%     [obj_0, grad] = q_mutual_inf(y, PHI, PHI_adj, PHI_C, PHI_C_adj);
% 
%     % Perform Blahut-Arimoto iteration
%     x_prev = x;
%     z_prev = z;
%     for t = 0:10000
%         z = expm(logm(z_prev) + grad / L_acc);
%         z = z / trace(z);
%         x = A_prev/A * x_prev + a/A * z;
%         
%         [obj_acc(k), grad_x] = q_mutual_inf(x, PHI, PHI_adj, PHI_C, PHI_C_adj);
%         [obj_y, ~] = q_mutual_inf(y, PHI, PHI_adj, PHI_C, PHI_C_adj);
%         if k == 1
%             obj_prev = obj_0;
%         else
%             obj_prev = obj_acc(k - 1);
%         end
%         if -obj_acc(k) <= -obj_y - trace(grad * (x - y)) + L_acc*quantum_relative_entropy(x, y) + a/(2*A)*epsilon
%             break
%         end
%         L_acc = L_acc * 1.1;
%         if L_acc > 1
%             L_acc = 1.0;
%             break
%         end        
%     end    
%     
% %     a = 1 / (theta^2);
% %     b = 1;
% %     c = -1;
% %     theta = (-b + sqrt(b^2 - 4*a*c)) / (2*a);
% 
%     % Adaptive step size heuristic
%     if k > 1
%     L_acc = trace(x * (grad - grad_x)) / quantum_relative_entropy(x, y);
%     if L_acc < 0 || L_acc > 1
%         L_acc = 1.0;
%     end    
%     end
%     if mu > L_acc
%         mu = L_acc;
%     end    
%     
%     obj_acc_z(k) = q_mutual_inf(z, PHI, PHI_adj, PHI_C, PHI_C_adj);
%     
%     if k >= 3
%     if obj_acc(k) < obj_acc(k-1)
%         A = 1/L;
%         z = x;
%     end
%     end    
% 
%     % Compute objective value
%     fprintf("Iteration: %d \t Objective: %.5e \t L: %.4f \t theta: %.4f\n", k, obj_acc(k), L_acc, theta)
%     
%     ub_acc(k) = real(max(eig(grad)));
% end


% Standard
rho_std = rho;
for k = 1:K
    [obj_std(k), grad] = q_mutual_inf(rho_std, PHI, PHI_adj, PHI_C, PHI_C_adj);

    % Perform Blahut-Arimoto iteration
    rho_std = expm(logm(rho_std) + 1/L * grad);
    rho_std = rho_std / trace(rho_std);

    % Compute objective value
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_std(k))
    
    ub_std(k) = real(max(eig(grad)));
end


semilogy(min([ub_heur; ub_std; ub_ls; ub_acc]) - obj_heur, 'r')
hold on
semilogy(min([ub_heur; ub_std; ub_ls; ub_acc]) - obj_std, 'g')
semilogy(min([ub_heur; ub_std; ub_ls; ub_acc]) - obj_ls, 'b')
semilogy(min([ub_heur; ub_std; ub_ls; ub_acc]) - obj_acc, 'k')

legend("Heur", "Std", "LS", "Acc")

figure
semilogy(ub_heur - max([obj_heur; obj_std; obj_ls; obj_acc]), 'r')
hold on
semilogy(ub_std - max([obj_heur; obj_std; obj_ls; obj_acc]), 'g')
semilogy(ub_ls - max([obj_heur; obj_std; obj_ls; obj_acc]), 'b')
semilogy(ub_acc - max([obj_heur; obj_std; obj_ls; obj_acc]), 'k')

legend("Heur", "Std", "LS", "Acc")

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


function [func, grad] = q_mutual_inf(rho, PHI, PHI_adj, PHI_C, PHI_C_adj)
    PHI_rho = ApplyMap(rho, PHI);
    PHI_C_rho = ApplyMap(rho, PHI_C);

    func = von_neumann_entropy(rho);
    func = func + von_neumann_entropy(PHI_rho);
    func = func - von_neumann_entropy(PHI_C_rho);

    grad = -logm(rho);
    grad = grad - ApplyMap(logm(PHI_rho), PHI_adj);
    grad = grad + ApplyMap(logm(PHI_C_rho), PHI_C_adj);
%     grad = grad - eye(size(rho));

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


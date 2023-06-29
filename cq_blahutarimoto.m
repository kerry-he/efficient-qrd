clear all; close all; clc;

n = 32; % Number of codewords 
N = 32; % Density matrix size

rng(10)
rho = zeros(N, N, n);
for i = 1:n
    rho(:, :, i) = rand_SPD(N);
    rho(:, :, i) = rho(:, :, i) / trace(rho(:, :, i));
%     rho(i, i, i) = 1;
end

p = ones(n, 1) / n;
% p = rand(n, 1);
% p = p / sum(p);

%% Setup algorithm
K = 80;
L = 0.03;
obj_ba = zeros(K, 1);
obj_acc = zeros(K, 1);
obj_acc2 = zeros(K, 1);
ub_ba = ones(K, 1)*Inf;
ub_acc = ones(K, 1)*Inf;
ub_acc2 = ones(K, 1)*Inf;

obj_ls = zeros(K, 1);
ub_ls = ones(K, 1)*Inf;


p_md = p;
p_ba = p;

alpha = zeros(n, 1);
% tic
L_1=L;
t_eval1 = zeros(K, 1);
for k = 1:K
    tic
    % Calculate ensemble density matrix
    Rho = zeros(N, N);
    for i = 1:n
        Rho = Rho + p_ba(i)*rho(:, :, i);
    end
    
    alpha_prev = alpha;
    p_prev = p_ba;

    % Perform Blahut-Arimoto iteration
    for i = 1:n
        alpha(i) = quantum_relative_entropy(rho(:, :, i), Rho);
    end
    for i = 1:n
        p_ba(i) = p_ba(i) * exp((alpha(i) - max(alpha)) / (L_1));
    end   
    
    p_ba = p_ba / sum(p_ba); % Normalise probability vector
    
    Rho_new = zeros(N, N);
    for i = 1:n
        Rho_new = Rho_new + p_ba(i)*rho(:, :, i);
    end
    
    L_1 = quantum_relative_entropy(Rho, Rho_new) / relative_entropy(p_ba, p_prev);
%     if L_1 < 0 || L_1 > 1
%         L_1 = 1.0;
%     end
    t_eval1(k) = toc;

    % Compute objective value
    obj_ba(k) = holevo_inf(p_ba, rho);
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_ba(k))
    
    ub_ba(k) = max(alpha);
end
% toc

% % Accelerated method
% tic
% x = p;
% z = p;
% obj_0 = holevo_inf(p, rho);
% theta = 1;
% Lacc = 0.001;
% for k = 1:K
%     y = (1 - theta) * x + theta * z;
%     % Calculate ensemble density matrix
%     Rho = zeros(N, N);
%     for i = 1:n
%         Rho = Rho + y(i)*rho(:, :, i);
%     end
%     
%     % Perform Blahut-Arimoto iteration
%     for i = 1:n
%         alpha(i) = quantum_relative_entropy(rho(:, :, i), Rho);
%     end
% %     for i = 1:n
% %         z(i) = z(i) * exp((alpha(i) - max(alpha)) / (theta*L));
% %     end
%     x_prev = x;
%     z_prev = z;
%     % Line search
%     for t = 0:100
%         for i = 1:n
%             z(i) = z_prev(i) * exp((alpha(i) - max(alpha)) / (theta*Lacc));
%         end  
%         z = z / sum(z); % Normalise probability vector
%         x = (1 - theta) * x_prev + theta * z;
% 
%         obj_acc(k) = holevo_inf(x, rho);
%         if k == 1
%             obj_prev = obj_0;
%         else
%             obj_prev = obj_acc(k - 1);
%         end
%         if k > 75
%             break
%         end
%         if -obj_acc(k) <= -obj_prev - (alpha' * (x - x_prev)) + Lacc*relative_entropy(x, x_prev)
%             break
%         end
%         Lacc = Lacc * 1.1;
%         
%         if Lacc > 1
%             Lacc = 1.0;
%             break
%         end
%     end    
% 
%     a = 1 / (theta^2);
%     b = 1;
%     c = -1;
%     theta = (-b + sqrt(b^2 - 4*a*c)) / (2*a);        
% %     theta = 2 / (k + 2);
%     
%     % Compute objective value
%     obj_acc(k) = holevo_inf(x, rho);
%     obj_acc_z(k) = holevo_inf(z, rho);
%     
%     if k >= 10
%     if obj_acc(k) < obj_acc(k-1)
%         theta = 1;
%         z = x;
%         Lacc = 0.001;
%     end
%     end
%     fprintf("Iteration: %d \t Objective: %.5e \t L: %.5f \t theta: %.4f\n", k, obj_acc(k), Lacc, theta)
%     
%     ub_acc(k) = max(alpha);
% end
% toc

% Line search
% tic
p_ls = p;
obj_0 = holevo_inf(p, rho);
Lk = 0.1;

t_eval = zeros(K, 1);

for k = 1:K
    tic
    % Calculate ensemble density matrix
    Rho = zeros(N, N);
    for i = 1:n
        Rho = Rho + p_ls(i)*rho(:, :, i);
    end
    
    alpha_prev = alpha;
    p_prev = p_ls;

    % Perform Blahut-Arimoto iteration
    for i = 1:n
        alpha(i) = quantum_relative_entropy(rho(:, :, i), Rho);
    end

    % Line search
    for t = 0:10000
        for i = 1:n
            p_ls(i) = p_prev(i) * exp(alpha(i) / Lk);
        end  
        p_ls = p_ls / sum(p_ls); % Normalise probability vector

        obj_ls(k) = holevo_inf(p_ls, rho);
        if k == 1
            obj_prev = obj_0;
        else
            obj_prev = obj_ls(k - 1);
        end
        if -obj_ls(k) <= -obj_prev - (alpha' * (p_ls - p_prev)) + Lk*relative_entropy(p_ls, p_prev)
%         if -obj_ls(k) <= -obj_prev - 1.001*(alpha' * (p_ls - p_prev))
            break
        end
        Lk = Lk * 2.0;
        if Lk > 1
            break
        end        
    end
    
    Rho_new = zeros(N, N);
    for i = 1:n
        Rho_new = Rho_new + p_ls(i)*rho(:, :, i);
    end
    
    Lk = (quantum_relative_entropy(Rho, Rho_new) + quantum_relative_entropy(Rho_new, Rho)) / (relative_entropy(p_ls, p_prev) + relative_entropy(p_prev, p_ls));
    if Lk <= 0 || Lk > 1
        Lk = 1e-5;
    end

    t_eval(k) = toc;
    
    % Compute objective value
%     obj_ls(k) = holevo_inf(p_ls, rho);
    fprintf("Iteration: %d \t Objective: %.5e \t L: %.5f\n", k, obj_ls(k), Lk)
    
    ub_ls(k) = obj_ls(k) - alpha' * p_ls + max(alpha);  
end
% toc

% Accelerated method
tic
x = p;
z = p;
theta = 1;
G = 1;
G_prev = 1;
G_min = 1e-3;
r = 1.5;
Lacc2 = 0.05;
% for k = 1:K
%     M = max(G / r, G_min);
%     
%     t = 0;
%     z_prev = z;
%     x_prev = x;
%     G_prev = G;
%     theta_prev = theta;
%     for t = 1:1000
%         G = M*r^(t - 1);
% 
%         if k >= 2
%             a = 1 / (G_prev*theta_prev^2);
%             b = 1 / G;
%             c = -1 / G;
%             theta = (-b + sqrt(b^2 - 4*a*c)) / (2*a);
%         end            
% 
%         y = (1 - theta) * x_prev + theta * z_prev;
%         % Calculate ensemble density matrix
%         Rho = zeros(N, N);
%         for i = 1:n
%             Rho = Rho + y(i)*rho(:, :, i);
%         end
% 
%         % Perform Blahut-Arimoto iteration
%         for i = 1:n
%             alpha(i) = quantum_relative_entropy(rho(:, :, i), Rho);
%             z(i) = z_prev(i) * exp(alpha(i) / (theta*Lacc2*G));
%         end
%         z = z / sum(z); % Normalise probability vector
%         x = (1 - theta) * x_prev + theta * z;
% %         for i = 1:n
% %             alpha(i) = quantum_relative_entropy(rho(:, :, i), Rho);
% %         end
% %         % Line search
% %         for cnt = 0:100
% %             for i = 1:n
% %                 z(i) = z_prev(i) * exp((alpha(i) - max(alpha)) / (theta*G*Lacc2));
% %             end  
% %             z = z / sum(z); % Normalise probability vector
% %             x = (1 - theta) * x + theta * z;
% % 
% %             obj_acc2(k) = holevo_inf(x, rho);
% %             if k == 1
% %                 obj_prev = obj_0;
% %             else
% %                 obj_prev = obj_acc2(k - 1);
% %             end
% %             if -obj_acc2(k) <= -obj_prev - (alpha' * (x - x_prev)) + Lacc2*relative_entropy(x, x_prev)
% %                 break
% %             end
% %             Lacc2 = Lacc2 * 1.1;
% % 
% %             if Lacc2 > 1
% %                 Lacc2 = 1.0;
% %                 break
% %             end
% %         end    
% %                 
%         if -holevo_inf(x, rho) <= -holevo_inf(y, rho) - alpha' * (x - y) + G*theta^2*Lacc2*relative_entropy(z, z_prev)
%             break
%         end
%     end
%     
% %     Rho_new = zeros(N, N);
% %     for i = 1:n
% %         Rho_new = Rho_new + p_ls(i)*rho(:, :, i);
% %     end
% %     
% %     Lacc2 = quantum_relative_entropy(Rho, Rho_new) / relative_entropy(y, x);
% %     if Lacc2 < 0 || Lacc2 > 1
% %         Lacc2 = 1.0;
% %     end    
%         
%     % Compute objective value
%     obj_acc2(k) = holevo_inf(x, rho);
%     obj_aaa(k) = holevo_inf(z, rho);
%     G_acc2(k) = G;
%     fprintf("Iteration: %d \t Objective: %.5e \t G: %.5f\n", k, obj_acc2(k), G)
%     
%     ub_acc2(k) = obj_acc2(k) - alpha' * x + max(alpha);
% end
for k = 1:K
    % Calculate ensemble density matrix
    Rho = zeros(N, N);
    for i = 1:n
        Rho = Rho + x(i)*rho(:, :, i);
    end
    

    % Perform Blahut-Arimoto iteration
    for i = 1:n
        alpha(i) = quantum_relative_entropy(rho(:, :, i), Rho);
    end
    for i = 1:n
        x(i) = x(i) * exp(alpha(i));
    end   
    
    x = x / sum(x); % Normalise probability vector

    % Compute objective value
    obj_acc2(k) = holevo_inf(x, rho);
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_acc2(k))
    
    ub_acc2(k) = max(alpha);   
end
toc

% Plot figures

semilogy(min([ub_ba; ub_acc; ub_acc2; ub_ls]) - obj_ba, 'r')
hold on
semilogy(min([ub_ba; ub_acc; ub_acc2; ub_ls]) - obj_acc2, 'b')
semilogy(min([ub_ba; ub_acc; ub_acc2; ub_ls]) - obj_ls, 'k')


legend("BA", "Acc", "Acc2", "ls")


% figure
% semilogy(ub_ba - max([obj_ba; obj_acc; obj_acc2; obj_ls]), 'r')
% hold on
% semilogy(ub_acc - max([obj_ba; obj_acc; obj_acc2; obj_ls]), 'g')
% semilogy(ub_acc2 - max([obj_ba; obj_acc; obj_acc2; obj_ls]), 'b')
% semilogy(ub_ls - max([obj_ba; obj_acc; obj_acc2; obj_ls]), 'k')
% 
% legend("BA", "Acc", "Acc2", "ls")




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

function out = holevo_inf_grad(p, rho)
    [N, ~, n] = size(rho);

    Rho = zeros(N, N);
    for i = 1:n
        Rho = Rho + p(i)*rho(:, :, i);
    end

    Rho_inv = Rho^-1;
    
    out = zeros(n, 1);
    for i = 1:n

        out(i) = quantum_relative_entropy(rho(:, :, i), Rho);
        for j = 1:n
            out(i) = out(i) - p(j) * trace(rho(:, :, j) * rho(:, :, i) * Rho_inv);
        end
    end
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
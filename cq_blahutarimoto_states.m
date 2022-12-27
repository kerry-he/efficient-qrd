clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))

rng(1)

n = 25; % Number of codewords 
N = 5; % Density matrix size

PHI = KrausOperators(RandomSuperoperator([N, N], 1, 0, 1));

PHI_adj = PHI;
for i = 1:length(PHI_adj)
    PHI_adj{i} = PHI_adj{i}';
end



%% Setup algorithm

% Initialise random states
p = rand(n, 1); p = p / sum(p);
rho = zeros(N, N, n);
for i = 1:n
    rho(:, :, i) = rand_SPD(N);
    rho(:, :, i) = rho(:, :, i) / trace(rho(:, :, i));
end

K = 1000;

% Alternative Nagaoka Blahut-Arimoto algorithm
p_my = p;
rho_my = rho;
L_1 = 1.0;
L_2 = 1.0;
obj_0 = holevo_inf(p_my, rho_my, PHI);

for k = 1:K
    % Previous values
    p_prev = p_my;
    rho_prev = rho_my;


    % Update probability distribution
    RHO = ensemble(p_my, rho_my);

    for i = 1:n
        grad(i) = quantum_relative_entropy(ApplyMap(rho_my(:, :, i), PHI), ApplyMap(RHO, PHI));
    end

    for t = 0:10000
        for i = 1:n
            p_my(i) = p_prev(i) * exp((grad(i) - max(grad)) / L_1);
        end   
        p_my = p_my / sum(p_my);

        obj_my(k) = holevo_inf(p_my, rho_my, PHI);
        if k == 1
            obj_prev = obj_0;
        else
            obj_prev = obj_my(k - 1);
        end
        if -obj_my(k) <= -obj_prev - (grad * (p_my - p_prev)) + L_1*relative_entropy(p_my, p_prev)
            break
        end
        L_1 = L_1 * 2.0;
        if L_1 > 1
            break
        end        
    end


    % Estimate smoothness
    RHO_prev = RHO;
    RHO = ensemble(p_my, rho_my);

    L_1 = quantum_relative_entropy(ApplyMap(RHO, PHI), ApplyMap(RHO_prev, PHI)) / relative_entropy(p_my, p_prev);
    if L_1 < 0.0 || L_1 > 1
        L_1 = 1.0;
    end


    % Update input states
    for i = 1:n
        grad_mtx(:, :, i) = ApplyMap(logm(ApplyMap(rho_my(:, :, i), PHI)) - logm(ApplyMap(RHO, PHI)), PHI_adj);
    end

    obj_prev = obj_my(k);

    for t = 0:10000
        for i = 1:n
            rho_my(:, :, i) = expm(logm(rho_prev(:, :, i)) + (grad_mtx(:, :, i) - eye(N)*max(diag(grad_mtx(:, :, i)))) / L_2);
            rho_my(:, :, i) = rho_my(:, :, i) / trace(rho_my(:, :, i)) + eye(N)*1e-10;
        end

        obj_my(k) = holevo_inf(p_my, rho_my, PHI);

        grad_x_rho = 0;
        avg_rel_entr = 0;
        for i = 1:n
            grad_x_rho = grad_x_rho + p_my(i)*trace(grad_mtx(:, :, i) * (rho_my(:, :, i) - rho_prev(:, :, i)));
            avg_rel_entr = avg_rel_entr + p_my(i)*quantum_relative_entropy(rho_my(:, :, i), rho_prev(:, :, i));
        end

        if -obj_my(k) <= -obj_prev - grad_x_rho + L_2*avg_rel_entr
            break
        end
        L_2 = L_2 * 2.0;
        if L_2 > 1.0
            break
        end        
    end        

    % Estimate smoothness
    RHO_prev = RHO;
    RHO = ensemble(p_my, rho_my);

    L_2 = quantum_relative_entropy(ApplyMap(RHO, PHI), ApplyMap(RHO_prev, PHI)) / quantum_relative_entropy(RHO, RHO_prev);
    if L_2 < 0.0 || L_2 > 1
        L_2 = 1.0;
    end    

    % Compute objective value
    fprintf("Iteration: %d \t Objective: %.5e \t L1: %.4f \t L2: %.4f\n", k, obj_my(k), L_1, L_2)
end


% Nagaoka Blahut-Arimoto algorithm
p_ba = p;
rho_ba = rho;

for k = 1:K
    % Calculate ensemble density matrix
    RHO = zeros(N, N);
    for i = 1:n
        RHO = RHO + p_ba(i)*rho_ba(:, :, i);
    end

    for i = 1:n
        temp_state = ApplyMap(logm(ApplyMap(rho_ba(:, :, i), PHI)) - logm(ApplyMap(RHO, PHI)), PHI_adj);
        B = p_ba(i) * expm(logm(rho_ba(:, :, i)) + temp_state);

        p_ba(i) = trace(B);
        rho_ba(:, :, i) = B / p_ba(i) + eye(N)*1e-15;
    end
    p_ba = p_ba / sum(p_ba);


    % Compute objective value
    obj_ba(k) = holevo_inf(p_ba, rho_ba, PHI);
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_ba(k))
end


% Alternative Nagaoka Blahut-Arimoto algorithm
p_ba2 = p;
rho_ba2 = rho;

for k = 1:K
    % Calculate ensemble density matrix
    RHO = zeros(N, N);
    for i = 1:n
        RHO = RHO + p_ba2(i)*rho_ba2(:, :, i);
    end

    for i = 1:n
        t_state = logm(ApplyMap(rho_ba2(:, :, i), PHI)) - logm(ApplyMap(RHO, PHI));
        temp_state = ApplyMap(t_state, PHI_adj);
        [U, D] = eig(temp_state);
        [D, idx] = sort(diag(D));
        U = U(:, idx);

        rho_ba2(:, :, i) = U(:, 1)*U(:, 1)' + eye(N)*1e-15;

        a = trace(rho_ba2(:, :, i) * t_state);
        p_ba2(i) = p_ba2(i) * exp(a);
    end
    p_ba2 = p_ba2 / sum(p_ba2);


    % Compute objective value
    obj_ba2(k) = holevo_inf(p_ba2, rho_ba2, PHI);
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_ba2(k))
end

semilogy(obj_my(end) - obj_my)
hold on
semilogy(obj_ba(end) - obj_ba)
semilogy(obj_ba2(end) - obj_ba2)


%% Functions
function RHO = ensemble(p, rho)
    [N, ~, n] = size(rho);
    RHO = zeros(N, N);
    for i = 1:n
        RHO = RHO + p(i)*rho(:, :, i);
    end        
end


function S = von_neumann_entropy(rho)
    S = -trace(rho * logm(rho));
end

function S = quantum_relative_entropy(rho, sigma)
    S = trace(rho * (logm(rho) - logm(sigma)));
end

function out = holevo_inf(p, rho, PHI)
    [N, ~, n] = size(rho);

    Rho = zeros(N, N);
    for i = 1:n
        rho(:, :, i) = ApplyMap(rho(:, :, i), PHI);
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
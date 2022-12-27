clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))

rng(2)
N = 3; % Density matrix size

A = RandomDensityMatrix(N, 1);
[~, A] = eig(A);
% U = RandomUnitary(N, 1);
% A = U * diag([0.5; 0.5; 0.0]) * U';
AR_vec = purification(A); AR = AR_vec * AR_vec';
% RA = PermuteSystems(AR, [2, 1]);

NR = length(AR) / N;

PHI = RandomSuperoperator([N, N], 1, 1, 1);
QRD = QRateDistortion(A);
% 
% % Monte Carlo
% L = 0;
% for i = 1:100
%     v = rand(N^2, 1); v = v / norm(v);
%     PHI_mc = v * v';
%     L_mc = norm(PartialTrace(PHI_mc, 1), 'fro');
%     if L_mc > L
%         L = L_mc;
%     end
% end
% alpha = 1;
% tau = 2 / (1 + sqrt(1+4/alpha*L^2));
% sigma = tau / alpha;
% fprintf("Monte Carlo L: %.2f\n", L)



%% Experiments
K = 500;
D = 0.0;
lambda = 0.0001;

% Change of variables
% RHO_QR = RandomDensityMatrix(N*NR, 1);
RHO_QR = kron(A, PartialTrace(AR, 1));
V_CV = -logm(PartialTrace(AR, 1));
% V_CV = zeros(N, N);
F_grad = AR;
% dF = eye(N^2, N^2);
dF = [];

tau = 0.5;
sigma = 30.0;
theta_0 = 1.01;

S_A = von_neumann_entropy(A);

tic
for k = 1:K
    obj_cv(k) = S_A + von_neumann_entropy(PartialTrace(RHO_QR, 2, N)) - von_neumann_entropy(RHO_QR);
    lgn_cv(k) = obj_cv(k) + lambda*(1 - AR_vec' * RHO_QR * AR_vec);
    GRAD_RHO = logm(RHO_QR) - kron(logm(PartialTrace(RHO_QR, 2, N)), eye(NR));
    V_prev = V_CV;
    RHO_QR_prev = RHO_QR;

    % Perform Blahut-Arimoto iteration
    X = logm(RHO_QR) - GRAD_RHO + lambda*F_grad;
    if k == 1
        N_hom = 1;
        for i = 1:N_hom
            lambda_temp = lambda * i / N_hom;
            X_temp = logm(RHO_QR) - GRAD_RHO + lambda_temp*F_grad;
            [V_CV, dF] = trace_preserve_V(X_temp, V_CV, AR, dF, true);
        end
    else
        [V_CV, dF] = trace_preserve_V(X, V_CV, AR, dF, true);
    end
    RHO_QR = expm(X - kron(eye(N), V_CV));

%     theta = theta_0 * 2;
%     tau_prev = tau;
%     sigma_prev = sigma;
% 
%     while true
% 
%         theta = theta / 2;
%         tau = tau_prev * theta;
%         sigma = sigma_prev * theta;
% 
%         STEP = -GRAD_RHO + lambda*F_grad - kron(eye(N), V_prev);
%         RHO_QR = expm(logm(RHO_QR_prev) + STEP*tau);
%         RHO_QR = RHO_QR / trace(RHO_QR);
%         
%         V_CV = V_prev + (PartialTrace(RHO_QR + theta*(RHO_QR - RHO_QR_prev), 1) - PartialTrace(AR, 1)) * sigma;    
%     
%         if trace((V_CV - V_prev)' * PartialTrace(RHO_QR - RHO_QR_prev, 1)) ...
%                 <= abs(quantum_relative_entropy(RHO_QR, RHO_QR_prev)) * (1 / tau - 1) + norm(V_CV - V_prev, 'fro')^2 / (2*sigma)
%             break
%         end
% 
%         display("BACKTRACKING")
% 
%     end
    

%     X = -GRAD_RHO + lambda*F_grad;
%     RHO_QR = expm(X);
%     P = (PartialTrace(RHO_QR, 1) \ PartialTrace(AR, 1))^(1/2);
%     RHO_QR = kron(eye(N), P) * RHO_QR * kron(eye(N), P);

%     lambda = max(lambda + 1.0*(1 - AR_vec' * (2*RHO_QR - RHO_QR_prev) * AR_vec - D), 0);
%     lambda_hist(k) = lambda;

    % Compute objective value
    fprintf("Iteration: %d \t Objective: %.5e \t lambda: %.5f\n", k, obj_cv(k), lambda)
end
toc

% Entropic Gradient Descent
PHI_EGD = RandomDensityMatrix(N^2, 1);%PHI;
V = zeros(N, N);

PHI_EGD_avg = zeros(N^2, N^2);
V_avg = zeros(N, N);

tic
for k = 1:K
%     obj_egd(k) = q_mutual_inf(A, RA, PHI_EGD);
%     GRAD_PHI = q_mutual_inf_grad(A, RA, PHI_EGD);
%     PHI_EGD = (PHI_EGD + PHI_EGD') / 2 + eye(N^2)*1e-11;
    PHI_EGD_prev = PHI_EGD;

    obj_egd(k) = QRD.q_mutual_inf(PHI_EGD);
    GRAD_PHI = QRD.q_mutual_inf_grad(PHI_EGD);    

    % Perform Blahut-Arimoto iteration
    X = logm(QRD.I_map(PHI_EGD)) - QRD.I_adj_inv_map(GRAD_PHI - lambda*QRD.F_grad);
    if k < 2
        V = QRD.trace_preserve_V(X, V, true);
    else
        V = QRD.trace_preserve_V(X, V, true);
    end    
    PHI_EGD = QRD.I_inv_map(expm(X - QRD.I_adj_inv_map(kron(V, eye(N)))));

%     STEP = QRD.I_adj_inv_map(GRAD_PHI - lambda*QRD.F_grad + kron(V, eye(N)));
%     PHI_EGD = QRD.I_inv_map(expm(logm(QRD.I_map(PHI_EGD)) - STEP * tau));
%     V = V + (PartialTrace(2*PHI_EGD - PHI_EGD_prev) - eye(N)) * sigma;

    % Update dual parameters
%     lambda = max(lambda + 0.5*(trace(C * PHI_EGD) - Cub), 0);
%     lambda_hist(k) = lambda;
    

    % Compute objective value
    fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_egd(k))
    ub(k) = min(eig(GRAD_PHI));
    
    PHI_EGD_avg = PHI_EGD_avg + PHI_EGD;
    V_avg = V_avg + V;    
end
toc
PHI_EGD_avg = PHI_EGD_avg/K;
V_avg = V_avg/K;    

% CVX Brute Force
ar = purification(A);
F_ub = 1 - ar' * RHO_QR * ar;
cvx_begin sdp
%         cvx_precision high
    variable rho(N^2, N^2) symmetric
    PHI_rho = QRD.I_map(rho);
%     PHI_rho = I_map(rho, AR, N, NR);
    minimize (quantum_entr(A) - quantum_cond_entr(PHI_rho, [N, NR], 2));
    rho >= 0;
%     for i = 1:N
%         for j = 1:N
%             trace(rho((i - 1)*N + 1:i*N, (j - 1)*N + 1:j*N)) == (i == j)*1;
%         end
%     end
    PartialTrace(rho, 2, N) == eye(N);
    1 - QRD.fidelity(rho) <= 0.43;
cvx_end
PHI_CVX = rho;
toc
% PHI_CVX = RandomDensityMatrix(N^2, 1);%PHI;
% PHI_rho_prev = I_map(PHI_CVX, RA);
% F_ub = 1 - fidelity(PHI_EGD, RA);
% tic
% for k = 1:K
%     obj_cvx(k) = QRD.q_mutual_inf(PHI_CVX);
%     GRAD_PHI = QRD.q_mutual_inf_grad(PHI_CVX);
% 
%     % Perform Blahut-Arimoto iteration
%     cvx_begin sdp quiet
% %         cvx_precision high
%         variable rho(N^2, N^2) symmetric
%         PHI_rho = QRD.I_map(rho);
%         minimize (trace(GRAD_PHI * rho) + quantum_rel_entr(PHI_rho, PHI_rho_prev));
%         rho >= 0;
%         for i = 1:N
%             for j = 1:N
%                 trace(rho((i - 1)*N + 1:i*N, (j - 1)*N + 1:j*N)) == (i == j)*1;
%             end
%         end
%         1 - fidelity(rho, RA) <= F_ub;
%     cvx_end
%     PHI_CVX = rho;
% 
%     % Compute objective value
%     fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_cvx(k))
%     ub(k) = min(eig(GRAD_PHI));
%     PHI_rho_prev = PHI_rho;
% end
% toc

% Projected Gradient Descent
% PHI_GD = PHI;
% for k = 1:K
%     obj_gd(k) = QRD.q_mutual_inf(PHI_GD);
%     GRAD_PHI = QRD.q_mutual_inf_grad(PHI_GD) - lambda*QRD.F_grad;
% 
%     % Perform Blahut-Arimoto iteration
%     % Gradient step
%     PHI_GD = PHI_GD - GRAD_PHI;
%     % Trace preserving step
%     for i = 1:N
%         for j = 1:N
%             if i ~= j
%                 Pij = PHI_GD((i - 1)*N + 1:i*N, (j - 1)*N + 1:j*N);
%                 Pij = Pij - trace(Pij) / N * eye(N);
%                 PHI_GD((i - 1)*N + 1:i*N, (j - 1)*N + 1:j*N) = Pij;
%             else
%                 Pij = PHI_GD((i - 1)*N + 1:i*N, (j - 1)*N + 1:j*N);
%                 Pij = Pij - (trace(Pij) - 1) / N * eye(N);
%                 PHI_GD((i - 1)*N + 1:i*N, (j - 1)*N + 1:j*N) = Pij;                
%             end
%         end 
%     end
%     % Completely positive step
%     [V,D] = eig(PHI_GD);
%     PHI_GD = V*max(real(D), 0)*V';
% 
%     % Compute objective value
%     fprintf("Iteration: %d \t Objective: %.5e\n", k, obj_gd(k))
% end

% semilogy(obj_egd - obj_egd(end))
% hold on
semilogy(obj_cv - obj_cv(end))

legend("EGD", "CVX")

%% Functions

function S = von_neumann_entropy(rho)
    S = -trace(rho * logm(rho + eye(length(rho))*1e-10));
end

function S = quantum_relative_entropy(rho, sigma)
    S = trace(rho * (logm(rho) - logm(sigma)));
end

function B = kron_choi(A, B)
    B = KrausOperators(B);
    [L, W] = size(B);
    for i = 1:L
        for j = 1:W
            B{i, j} = kron(A, B{i, j});
        end
    end   
end

function func = entr_exch(RA, PHI)
    I_PHI = kron_choi(sqrt(length(RA)), PHI);
    RB = ApplyMap(RA, PHI);
end

function func = q_mutual_inf(A, RA, PHI)
    N = length(A);
%     PHI_C = ComplementaryMap(PHI, [N, N]);
%     
%     PHI_A = ApplyMap(A, PHI);
%     PHI_C_A = ApplyMap(A, PHI_C);
% 
%     func = von_neumann_entropy(A);
%     func = func + von_neumann_entropy(PHI_A);
%     func = func - von_neumann_entropy(PHI_C_A);
    
    func = von_neumann_entropy(A);
    func = func + von_neumann_entropy(i_map(PHI, A));
    func = func - von_neumann_entropy(I_map(PHI, RA));    
end

% function GRAD_PHI = q_mutual_inf_grad(A, RA, PHI)
%     N = length(A);
%     
%     log_PHI_A = logm(ApplyMap(A, PHI));
%     log_IPHI_RA = logm(ApplyMap(RA, kron_choi(eye(N), PHI)) + 1e-10*eye(N^2));
%     
%     GRAD_PHI = zeros(size(PHI));
%     for i = 1:N
%         for j = 1:N
%             GRADij = -A(i, j) * (log_PHI_A + eye(N));
%             
%             for x = 1:N
%                 for y = 1:N
%                     RA_ijxy = RA((x - 1)*N + i, (y - 1)*N + j);
%                     Ix = zeros(N*N, N); Iy = Ix;
%                     Ix((x - 1)*N + 1:x*N, :) = eye(N);
%                     Iy((y - 1)*N + 1:y*N, :) = eye(N);
%                     
%                     GRADij = GRADij + RA_ijxy * Iy' * (log_IPHI_RA + eye(N^2)) * Ix;
%                 end
%             end
%             
%             GRAD_PHI((i - 1)*N + 1:i*N, ...
%                      (j - 1)*N + 1:j*N) = GRADij';
%         end 
%     end
% end

function GRAD_PHI = q_mutual_inf_grad(A, RA, PHI)
    N = length(A);
    GRAD_PHI = I_map_adj(logm(I_map(PHI, RA)) + eye(N^2), RA);
    GRAD_PHI = GRAD_PHI - i_map_adj(logm(i_map(PHI, A)) + eye(N), A);
end

function out = I_map(PHI, RA, N, NR)
    out = zeros(N*NR, N*NR);
    for i = 1:NR
        for j = 1:NR
            Ii = zeros(N*NR, N); Ij = Ii;
            Ii((i - 1)*N + 1:i*N, :) = eye(N);
            Ij((j - 1)*N + 1:j*N, :) = eye(N);            
            Pij = Ii' * PHI * Ij;            
            
            for x = 1:NR
                for y = 1:NR
                    RA_ijxy = RA((i - 1)*N + x, (j - 1)*N + y);
                    Ix = zeros(N*NR, N); Iy = Ix;

                    xi = x:NR:N*NR; xj = 1:N;
                    for k = 1:N
                        Ix(xi(k), xj(k)) = 1;
                    end
                    yi = y:NR:N*NR; yj = 1:N;
                    for k = 1:N
                        Iy(yi(k), yj(k)) = 1;
                    end
                    
                    out = out + Ix * Pij * Iy' * RA_ijxy;
                end
            end
        end 
    end
end

function out = I_map_adj(PHI, RA)
    N = sqrt(length(RA));
    out = zeros(N^2, N^2);
    for i = 1:N
        for j = 1:N
            Ii = zeros(N*N, N); Ij = Ii;
            Ii((i - 1)*N + 1:i*N, :) = eye(N);
            Ij((j - 1)*N + 1:j*N, :) = eye(N);            
            
            for x = 1:N
                for y = 1:N
                    RA_ijxy = RA((x - 1)*N + i, (y - 1)*N + j);
                    Ix = zeros(N*N, N); Iy = Ix;
                    Ix((x - 1)*N + 1:x*N, :) = eye(N);
                    Iy((y - 1)*N + 1:y*N, :) = eye(N);
                    
                    out = out + Ij * Iy' * PHI * Ix * Ii' * RA_ijxy;
                end
            end
        end 
    end
end

function PHI_INV = I_map_inv(PHI, RA)
    N = sqrt(length(RA));
    K = zeros(N^4, N^4);
    for i = 1:N
        for j = 1:N
            Ii = zeros(N*N, N); Ij = Ii;
            Ii((i - 1)*N + 1:i*N, :) = eye(N);
            Ij((j - 1)*N + 1:j*N, :) = eye(N);
            
            for x = 1:N
                for y = 1:N
                    RA_ijxy = RA((x - 1)*N + i, (y - 1)*N + j);
                    Ix = zeros(N*N, N); Iy = Ix;
                    Ix((x - 1)*N + 1:x*N, :) = eye(N);
                    Iy((y - 1)*N + 1:y*N, :) = eye(N);
                    
                    K = K + kron(Ix*Ii', Iy*Ij') * RA_ijxy;
                end
            end
        end 
    end
    
    PHI_INV = K \ PHI(:);
    PHI_INV = reshape(PHI_INV, [N^2, N^2]);
end

function PHI_INV = I_map_adj_inv(PHI, RA)
    N = sqrt(length(RA));
    K = zeros(N^4, N^4);
    for i = 1:N
        for j = 1:N
            Ii = zeros(N*N, N); Ij = Ii;
            Ii((i - 1)*N + 1:i*N, :) = eye(N);
            Ij((j - 1)*N + 1:j*N, :) = eye(N);
            
            for x = 1:N
                for y = 1:N
                    RA_ijxy = RA((x - 1)*N + i, (y - 1)*N + j);
                    Ix = zeros(N*N, N); Iy = Ix;
                    Ix((x - 1)*N + 1:x*N, :) = eye(N);
                    Iy((y - 1)*N + 1:y*N, :) = eye(N);
                    
                    K = K + kron(Ij*Iy', Ii*Ix') * RA_ijxy;                    
                end
            end
        end 
    end
    
    PHI_INV = K \ PHI(:);
    PHI_INV = reshape(PHI_INV, [N^2, N^2]);
end

function CHOI = trace_preserve_X(X)
    n = 1e10;
    X = expm(X / n)^n;

    N = sqrt(length(X));
    P = PartialTrace(X, 2);
    P_inv_sqrt = P^(-0.5);
    I = eye(N);
    CHOI = kron(P_inv_sqrt, I) * X * kron(P_inv_sqrt, I)'; 
end

function CHOI = trace_preserve(CHOI)
	N = sqrt(length(CHOI));
%     KRAUS = KrausOperators(CHOI);
%     
%     P = zeros(N, N);
%     for i = 1:length(KRAUS)
%         P = P + KRAUS{i}' * KRAUS{i};
%     end
%     
%     P_sqrt_inv = P^(-1/2);
%     for i = 1:length(KRAUS)
%         KRAUS{i} = KRAUS{i} * P_sqrt_inv;
%     end
%     
%     CHOI = ChoiMatrix(KRAUS);

    P = PartialTrace(CHOI, 2);
    P_inv_sqrt = P^(-0.5);
    I = eye(N);
    A = RandomUnitary(N, 1);
    CHOI = kron(P_inv_sqrt, I) * CHOI * kron(P_inv_sqrt, I)';    
end

function KRAUS = choi_to_kraus(CHOI)
    N = length(CHOI);
    [V, D] = eig(CHOI);
    
    KRAUS = cell(N, 1);
    for i = 1:N
        KRAUS{i} = reshape(V(:, i), [sqrt(N), sqrt(N)]) * sqrt(D(i, i));
    end
end

function out = fidelity(PHI, RA, N, NR)
    [V, D] = eig(RA);
    RA_vec = zeros(length(V), 1);
    for i = 1:length(D)
        RA_vec = RA_vec + V(:, i) * D(i, i);
    end

    out = 0;
    for i = 1:NR
        for j = 1:NR
            Ii = zeros(N*NR, N); Ij = Ii;
            Ii((i - 1)*N + 1:i*N, :) = eye(N);
            Ij((j - 1)*N + 1:j*N, :) = eye(N);            
            Pij = Ii' * PHI * Ij;            
            
            for x = 1:NR
                for y = 1:NR
                    RA_ijxy = RA((i - 1)*N + x, (j - 1)*NR + y);
                    Ix = zeros(N*NR, N); Iy = Ix;

                    xi = x:NR:N*NR; xj = 1:N;
                    for k = 1:N
                        Ix(xi(k), xj(k)) = 1;
                    end
                    yi = y:NR:N*NR; yj = 1:N;
                    for k = 1:N
                        Iy(yi(k), yj(k)) = 1;
                    end
                    
                    out = out + RA_vec' * Ix * Ii' * PHI * Ij * Iy' * RA_vec * RA_ijxy;
                end
            end
        end 
    end
end

function out = fidelity_grad(RA)
    [V, D] = eig(RA);
    RA_vec = zeros(length(V), 1);
    for i = 1:length(D)
        RA_vec = RA_vec + V(:, i) * D(i, i);
    end

    N = sqrt(length(RA));
    out = zeros(N^2, N^2);
    for i = 1:N
        for j = 1:N
            Ii = zeros(N*N, N); Ij = Ii;
            Ii((i - 1)*N + 1:i*N, :) = eye(N);
            Ij((j - 1)*N + 1:j*N, :) = eye(N);            
            
            for x = 1:N
                for y = 1:N
                    RA_ijxy = RA((x - 1)*N + i, (y - 1)*N + j);
                    Ix = zeros(N*N, N); Iy = Ix;
                    Ix((x - 1)*N + 1:x*N, :) = eye(N);
                    Iy((y - 1)*N + 1:y*N, :) = eye(N);
                    
                    out = out + (RA_vec' * Ix * Ii')' * (Ij * Iy' * RA_vec)' * RA_ijxy;
                end
            end
        end 
    end
end

function [V, dF] = trace_preserve_V(X, V, AR, dF, bfgs)
    NR = length(V);
    N = length(X) / NR;
    R = PartialTrace(AR, 1, N);

    F = PartialTrace(expm(X - kron(eye(N), V)), 1, N) - R;
    F = F(:);
    nF = norm(F);

    while true


        if norm(F) < 1e-10
            break
        end
        
        if isempty(dF)
            tic            
            dF = zeros(NR^2, NR^2);

            [Df, Vf] = exp_fdd(X - kron(eye(N), V));

            for i = 1:NR
                for j = 1:NR
                    if i > j
                        temp = reshape(dF((j - 1)*NR + i, :), [N, N]);
                        temp = temp';
                        dF((i - 1)*NR + j, :) = temp(:);
                    else
                        H = sparse(i, j, 1, NR, NR);
                        dFdH = PartialTrace(fdd(Df, Vf, -kron(eye(N), H)), 1, N);        
                        dF((i - 1)*NR + j, :) = dFdH(:);
                    end
                end
            end
            fprintf("Calculated Jacobian in: %.5f \n", toc)

            dF = inv(dF);
        end
   
        % Compute step direction
        p = -dF * F;
%         p = -dF \ F;

    
        % Line search
        alpha = 1.0;
        beta = -1;
        for i = 1:1
            s = alpha * p;
            V_vec = V(:) + s;
            V_new = reshape(V_vec, size(V));
%             V = (V + V') / 2;

            F_new = PartialTrace(expm(X - kron(eye(N), V_new)), 1, N) - R;
            F_new = F_new(:);
            
            if norm(F_new) <= norm(F)
                break
            end
            alpha = alpha * beta;
        end
        
        V = V_new;
        
        y = F_new - F;
        rho = 1 / (y'*s);
        if ~bfgs
            dF = [];
            fprintf("\t Error: %.5e\n", norm(F_new))
        else
            dF = (eye(NR^2) - rho*s*y') * dF * (eye(NR^2) - rho*y*s') + rho*(s*s');
%             dF = dF + (s'*y + y'*dF*y)*(s*s')/(s'*y)^2 - (dF*y*s' + s*y'*dF)/(s'*y);
            fprintf("\t Error: %.5e\n", norm(F_new))
        end

        F = F_new;
%         V_vec = V(:) - (dF \ F);
%         V = reshape(V_vec, size(V));
%         V = (V + V') / 2;
        
%         break
    end
end

function [Df, V] = exp_fdd(A)
    [V, D] = eig(A);
    N = length(A);

    Df = zeros(N, N);

    for i = 1:N
        for j = 1:N
            if i == j || D(i, i) == D(j, j)
                Df(i, j) = exp(D(i, j));
            else
                Df(i, j) = (exp(D(i, i)) - exp(D(j, j))) / (D(i, i) - D(j, j));
            end
        end
    end

end


function Df = fdd(Df, V, H)
    Df = Df .* (V'*H*V);
    Df = V * Df * V';
end

function out = i_map(PHI, A)
    N = length(A);
    out = zeros(N, N);
    for i = 1:N
        for j = 1:N
            Ii = sparse(N*N, N); Ij = Ii;
            Ii((i - 1)*N + 1:i*N, :) = eye(N);
            Ij((j - 1)*N + 1:j*N, :) = eye(N);            
            out = out + Ii' * PHI * Ij * A(i, j);
        end 
    end    
end

function out = i_map_adj(PHI, A)
    N = length(A);
    out = zeros(N^2, N^2);
    for i = 1:N
        for j = 1:N
            Ii = sparse(N*N, N); Ij = Ii;
            Ii((i - 1)*N + 1:i*N, :) = eye(N);
            Ij((j - 1)*N + 1:j*N, :) = eye(N);
            out = out + Ij * PHI * Ii' * A(i, j);
        end 
    end    
end
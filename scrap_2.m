clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))

% rng(1)
N = 4;
A = RandomDensityMatrix(N, 1); [~, A] = eig(A);
B = RandomDensityMatrix(N, 1);
C = RandomDensityMatrix(N, 1);
D = RandomDensityMatrix(N, 1);
E = RandomDensityMatrix(N, 1);
F = RandomDensityMatrix(N^2, 1);

U = RandomUnitary(N, 1);
V = RandomUnitary(N, 1);

AR_vec = purification(A); AR = AR_vec * AR_vec';
R = PartialTrace(AR, 1);

I = eye(N);
Z = zeros(N);

RHO = RandomDensityMatrix(N, 1);

PHI = RandomSuperoperator(N, 1, 0, 1);
KRAUS = KrausOperators(PHI);


m = 4;      % Number of inputs
n = 4;      % Number of outputs
% lambda = 1.0;
delta = rand(n, m);
% q = rand(n, 1); q = q / sum(q);
% Q = q .* exp(-lambda * delta); Q = Q ./ sum(Q);
% 
% X = [q .* exp(-lambda * delta), q*0];
% X = X';
% 
% NS = null(X);
% c = ones(n, 1);
% c = c + NS(:, 1)*1.0;
% 
% b = c .* q; % b = b ./ sum(b); 
% % b = rand(n, 1); b = b / sum(b);
% P = b .* exp(-lambda * delta); P = P ./ sum(P);
% 
% LHS = 0;
% p = rand(m, 1); p = p / sum(p);
% RHS = relative_entropy(Q*p, P*p);
% for i = 1:n
%     for j = 1:m
%         LHS = LHS + p(j) * relative_entropy(Q(i, j), P(i, j));
%     end
% end

% delta = 1 - eye(m);
% lambda = 5.0;
% p = rand(m, 1); p = p / sum(p);
% q = rand(n, 1); q = q / sum(q);
% Q = q .* exp(-lambda * delta); Q = Q ./ sum(Q);
% b = rand(n, 1); b = b / sum(b);
% b = q + rand(n, 1) * 1e-5; b = b / sum(b);
% P = b .* exp(-lambda * delta); P = P ./ sum(P);
% 
% 
% NUM = relative_entropy(Q*p, P*p);
% DEN = 0;
% for i = 1:n
%     for j = 1:m
%         DEN = DEN + p(j) * relative_entropy(Q(i, j), P(i, j));
%     end
% end
% NUM/DEN

m = 4;      % Number of inputs
n = 4;      % Number of outputs
Q = rand(n, m);
Q = Q ./ sum(Q);
P0 = rand(n, m);
P0 = P0 ./ sum(P0);

p = rand(m, 1); p = p / sum(p);
t = linspace(0, 1, 10000);

for k = 1:9999
    P = P0 + t(k) * (Q - P0);

    NUM = relative_entropy(P*p, Q*p);
    DEN = 0;
    for i = 1:n
        for j = 1:m
            DEN = DEN + p(j) * relative_entropy(P(i, j), Q(i, j));
        end
    end
    ratio(k) = NUM/DEN;
    
    num(k) = NUM;
    den(k) = DEN;

end

plot(ratio)

%% Functions

function out = tensor_choi_map(A, B, N)
    I = eye(N);
    out = kron(A, I) * kron(I, B);
end

function S = von_neumann_entropy(rho)
    S = -trace(rho * logm(rho));
end

function S = quantum_relative_entropy(rho, sigma)
    S = trace(rho * (logm(rho) - logm(sigma)));
end

function H = relative_entropy(p, q)
    H = 0;
    for i = 1:length(p)
        H = H + p(i) * log(p(i) / q(i));
    end
end

function A = rand_SPD(n)
    A = rand(n);
    A = A * A';
end

function func = q_mutual_inf(PHI, A, RA)
    N = length(A);
    
    PHI_C = ComplementaryMap(PHI, [N, N]);

    PHI_rho = ApplyMap(A, PHI);
    PHI_C_rho = ApplyMap(A, PHI_C);

    func = von_neumann_entropy(A);
    func = func + von_neumann_entropy(PHI_rho);
    func = func - von_neumann_entropy(PHI_C_rho);
    
%     tic
%     func = von_neumann_entropy(A);
%     func = func + von_neumann_entropy(i_map(PHI, A));
%     func = func - von_neumann_entropy(I_map(PHI, RA));
%     toc
end

function GRAD_PHI = q_mutual_inf_grad(PHI, A, RA)
    N = length(A);
    GRAD_PHI = I_map_adj(logm(I_map(PHI, RA)) + eye(N^2), RA);
    GRAD_PHI = GRAD_PHI - i_map_adj(logm(i_map(PHI, A)) + eye(N), A);
end

function func = entr_exch(rho, PHI, N)
    PHI_C = ComplementaryMap(PHI, [N, N]);

    PHI_C_rho = ApplyMap(rho, PHI_C);

    func = von_neumann_entropy(PHI_C_rho);
end

function out = partial_trace(AB, sys)
    N = sqrt(length(AB));
    out = zeros(N, N);

    if sys == 1
        for i = 1:N
            I = zeros(N*N, N);
            I((i - 1)*N + 1:i*N, :) = eye(N);

            out = out + I' * AB * I;
        end        
    else
        for i = 1:N
            I = zeros(N*N, N);
            x = i:N:N^2; y = 1:N;
            for j = 1:N
                I(x(j), y(j)) = 1;
            end

            out = out + I' * AB * I;
        end            
    end
end

function out = kroneker(A, sys)
    N = length(A);
    out = zeros(N^2, N^2);

    if sys == 1
        for i = 1:N
            I = zeros(N*N, N);
            I((i - 1)*N + 1:i*N, :) = eye(N);

            out = out + I * A * I';
        end        
    else
        for i = 1:N
            I = zeros(N*N, N);
            x = i:N:N^2; y = 1:N;
            for j = 1:N
                I(x(j), y(j)) = 1;
            end

            out = out + I * A * I';
        end            
    end
end

function CHOI = trace_preserve(X)
    n = 1e5;
    X = expm(X / n)^n;

    N = sqrt(length(X));
    P = PartialTrace(X, 2);
    P_inv_sqrt = P^(-0.5);
    I = eye(N);
    CHOI = kron(P_inv_sqrt, I) * X * kron(P_inv_sqrt, I)'; 
end

function out = fidelity(PHI, RA)
    [V, D] = eig(RA);
    RA_vec = zeros(length(V), 1);
    for i = 1:length(D)
        RA_vec = RA_vec + V(:, i) * D(i, i);
    end

    N = sqrt(length(RA));
    out = 0;
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

function V = trace_preserve_V(X, V)
    N = length(V);

    for t = 1:100
        F = PartialTrace(expm(X - kron(V, eye(N))), 2) - eye(N);
        F = F(:);

        if norm(F) < 1e-5
            break
        end
    
        dF = zeros(N^2, N^2);
        
        for i = 1:N
            for j = 1:N
                H = zeros(N);
                H(i, j) = 1;
                dFdH = PartialTrace(exp_fdd(X - kron(V, eye(N)), -kron(H, eye(N))), 2);
        
                dF((i - 1)*N + j, :) = dFdH(:);
            end
        end
    
        V_vec = V(:) - dF \ F;
        V = reshape(V_vec, size(V));
    end
end

% function out = I_map(PHI, RA)
%     N = sqrt(length(RA));
%     out = zeros(N^2, N^2);
%     for i = 1:N
%         for j = 1:N
%             Pij = PHI((i - 1)*N + 1:i*N, (j - 1)*N + 1:j*N);
%             
%             for x = 1:N
%                 for y = 1:N
%                     RA_ijxy = RA((i - 1)*N + x, (j - 1)*N + y);
%                     Ix = zeros(N*N, N); Iy = Ix;
% 
%                     xi = x:N:N^2; xj = 1:N;
%                     for k = 1:N
%                         Ix(xi(k), xj(k)) = 1;
%                     end
%                     yi = y:N:N^2; yj = 1:N;
%                     for k = 1:N
%                         Iy(yi(k), yj(k)) = 1;
%                     end
%                     
%                     out = out + Ix * Pij * Iy' * RA_ijxy;
%                 end
%             end
%         end 
%     end
% end
% 
% function PHI_INV = I_map_inv(PHI, RA)
%     N = sqrt(length(RA));
%     K = zeros(N^4, N^4);
%     for i = 1:N
%         for j = 1:N
%             Ii = zeros(N*N, N); Ij = Ii;
%             Ii((i - 1)*N + 1:i*N, :) = eye(N);
%             Ij((j - 1)*N + 1:j*N, :) = eye(N);
%             
%             for x = 1:N
%                 for y = 1:N
%                     RA_ijxy = RA((i - 1)*N + x, (j - 1)*N + y);
%                     Ix = zeros(N*N, N); Iy = Ix;
% 
%                     xi = x:N:N^2; xj = 1:N;
%                     for k = 1:N
%                         Ix(xi(k), xj(k)) = 1;
%                     end
%                     yi = y:N:N^2; yj = 1:N;
%                     for k = 1:N
%                         Iy(yi(k), yj(k)) = 1;
%                     end
%                     
%                     K = K + kron(Ix*Ii', Iy*Ij') * RA_ijxy;
%                 end
%             end
%         end 
%     end
%     
%     PHI_INV = inv(K) * vec(PHI);
%     PHI_INV = reshape(PHI_INV, [N^2, N^2]);
% end
% 

function out = I_map(PHI, RA)
    N = sqrt(length(RA));
    out = zeros(N^2, N^2);
    for i = 1:N
        for j = 1:N
            Ii = sparse(N*N, N); Ij = Ii;
            Ii((i - 1)*N + 1:i*N, :) = eye(N);
            Ij((j - 1)*N + 1:j*N, :) = eye(N);            
            Pij = Ii' * PHI * Ij;            
            
            for x = 1:N
                for y = 1:N
                    RA_ijxy = RA((x - 1)*N + i, (y - 1)*N + j);
                    Ix = sparse(N*N, N); Iy = Ix;
                    Ix((x - 1)*N + 1:x*N, :) = eye(N);
                    Iy((y - 1)*N + 1:y*N, :) = eye(N);
                    
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
clear all; close all; clc;
addpath(genpath('QETLAB-0.9'))
addpath(genpath('quantinf'))
addpath(genpath('cvxquad-master'))

rng(1)
n = 2;
A = RandomDensityMatrix(n, 1);
[~, A] = eig(A);

idx = 1:n+1:n^2;
sp = 1:n^2;
for i = idx(end:-1:1)
    sp(i) = [];
end

tr_b_idx = zeros(n, n-1);
tr_r_idx = zeros(n, n-1);
for i=1:n
    temp = i:n:n^2;
    temp(i) = [];
    tr_b_idx(i, :) = temp;

    temp = (i-1)*n+1:i*n;
    temp(i) = [];
    tr_r_idx(i, :) = temp;
end

tr_b_idx = tr_b_idx - (ceil(tr_b_idx ./ (n+1)));
tr_r_idx = tr_r_idx - (ceil(tr_r_idx ./ (n+1)));

AR_vec = sparse(1:n+1:n^2, ones(n, 1), sqrt(diag(A)));
AR = AR_vec * AR_vec';

AR_beta = sqrt(diag(A)) * sqrt(diag(A))';

R = PartialTrace(AR, 1, n); 

I = speye(n);

% Construct permutation matrix
sp_idx = zeros(n^2, 1);
alpha = 1; beta = n^2 - n + 1; count = 1;
for i = 1:n
    for j = 1:n
        if i ~= j
            sp_idx(count) = alpha;
            alpha = alpha + 1;
        else
            sp_idx(count) = beta;
            beta = beta + 1;
        end
        count = count + 1;
    end
end
P = sparse(sp_idx, 1:n^2, ones(n^2, 1), n^2, n^2);

% Construct sparse indexing
k=n^2-n;
i_sp = zeros(k + n^2, 1); j_sp = i_sp; % Make sparse structure for U
i_sp(1:k) = 1:k; j_sp(1:k) = 1:k;
cnt = k + 1; M = n^2 - k;
for i = k+1:n^2
    i_sp(cnt:cnt+M-1) = k+1:n^2;
    j_sp(cnt:cnt+M-1) = i;
    cnt = cnt + M;
end
idx_sp = [i_sp, j_sp];

lambda = 1;

%%
% tic
% cvx_begin sdp
%     variable alpha(n^2-n)
%     variable beta(n, n) symmetric
%     variable rho_b(n)
%     minimize ( quantum_entr(A) + quantum_rel_entr( beta, diag(rho_b) ) ...
%         + sum(rel_entr( alpha, reshape(repmat(rho_b, 1, n-1)', [n^2-n, 1]) )) ...
%         - lambda * ( sum(sum(beta .* AR_beta)) ) );
%     alpha >= 0;
%     beta >= 0;
% 
%     for i = 1:n
%         beta(i, i) + sum(alpha(tr_b_idx(i, :))) == R(i, i);
%         rho_b(i) == beta(i, i) + sum(alpha(tr_r_idx(i, :)));
%     end
% cvx_end
% toc
% 
% rho = rcstr(alpha, beta, P, idx_sp)




tic
cvx_begin sdp
    variable rho(n^2, n^2) symmetric
    minimize (quantum_entr(A) - quantum_cond_entr(rho, [n, n], 2) - lambda * trace(rho * AR));
    rho >= 0;
    PartialTrace(rho, 1, n) == R;
cvx_end
toc


%% Reconstruct matrix
function rho = rcstr(alpha, beta, P, idx_sp)
    n = length(beta);

    v_sp = zeros(n^2-n + n^2, 1);
    v_sp(1:n^2-n) = alpha;
    v_sp(n^2-n+1:end) = beta(:);
    rho = sparse(idx_sp(:, 1), idx_sp(:, 2), v_sp);

    rho = P' * rho * P;
end
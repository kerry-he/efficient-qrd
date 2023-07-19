function [BR_feas, BR_D_feas, BR, V] = solveEfQrdSub(BR, V, AR, kappa, opt)
    %SOLVEQRDEFSUB Solves the mirror descent subproblem for the quantum
    %rate-distortion problem with entanglement fidelity distortion.

    N = sqrt(length(BR));
    R = sparsePartialTrace(AR, 1);
    I = speye(N);

    % Compute primal iterate
    X_prev = kron(spdiag(log(sparsePartialTrace(BR, 2))), I);
    X = X_prev - kron(I, V) + kappa*AR;
    [BR_U, D] = sparseEig(X);
    BR_D = exp(D);
    BR = BR_U * spdiag(BR_D) * BR_U';

    % Compute feasible primal iterate
    BR_feas = proj(BR, R);
    [BR_U_feas, BR_D_feas] = sparseEig(BR_feas);

    % Function evaluation
    obj = -trace(BR) - sum(R .* diag(V));
    grad = sparsePartialTrace(BR, 1) - R;

    % Optimality gap for inexact stopping criterion
    gap = sparseQRE(BR_D_feas, BR_U_feas, BR_D, BR_U) ...
        - trace(BR_feas) + trace(BR);
    
    if opt.verbose == 2
        fprintf("|  %d  \t  %.1e \n", 0, gap)
    end    

    k = 1;
    while gap >= opt.sub.tol

        % Compute ascent direction
        if strcmp(opt.sub.alg, 'gradient')
            p = grad;
        elseif strcmp(opt.sub.alg, 'newton')
            Hess = getHessian(D, BR_U);
            p = -Hess \ grad;
        elseif strcmp(opt.sub.alg, 'cg')
            Df = getFddMatrix(D);
            hessProd = @(H) sparsePartialTrace(...
                ddTrFunc(Df, BR_U, kron(I, spdiag(H))), 1);
            [p, ~] = cgs(hessProd, grad);
        else
            error(['Option sub.alg is invalid ' ...
                '(must be ''gradient'', ''newton'', or ''cg'')'])
        end

        t = opt.sub.t0;

        while true
            % Update dual variable
            V_new = V + t*spdiag(p);
    
            % Compute primal iterate
            X = X_prev - kron(I, V_new) + kappa*AR;
            [BR_U, D] = sparseEig(X);
            BR_D = exp(D);
            BR = BR_U * spdiag(BR_D) * BR_U';

            % Function evaluation
            obj_new = -trace(BR) - sum(R .* diag(V_new));

            % Backtracking line search
            if -obj_new <= -obj - opt.sub.alpha*t*(grad' * p)
                break
            end
            t = t * opt.sub.beta;
        end

        % Exit if no progress is being made
        if obj - obj_new == 0
            break
        end

        grad = sparsePartialTrace(BR, 1) - R;
        obj = obj_new;
        V = V_new;

        % Compute feasible primal iterate
        BR_feas = proj(BR, R);
        [BR_U_feas, BR_D_feas] = sparseEig(BR_feas);
        
        % Optimality gap for inexact stopping criterion
        gap = sparseQRE(BR_D_feas, BR_U_feas, BR_D, BR_U) ...
            - trace(BR_feas) + trace(BR);
    
        if opt.verbose == 2
            fprintf("                        |                 |  %d  \t  %.1e \n", ...
                k, gap)
        end    
        k = k + 1;

    end

end

%% Auxiliary functions

function BR_feas = proj(rho, R)
    N = length(R);
    I = speye(N);
    proj_op = (R ./ sparsePartialTrace(rho, 1)).^0.5;
    proj_op = spdiag(proj_op);

    BR_feas = kron(I, proj_op) * rho * kron(I, proj_op');
end

function J = getHessian(D, U)
    N = sqrt(length(D));

    J = zeros(N, N);
    Df = getFddMatrix(D);

    for i = 1:N
        H = sparse(i:N:N^2, i:N:N^2, -1, N^2, N^2);
        J_temp = ddTrFunc(Df, U, H);
        J(i, :) = sparsePartialTrace(J_temp, 1, N);
    end
end

function Df = getFddMatrix(D)

    N = sqrt(length(D));

    blk_idx = 1:N+1:N^2;
    dgl_idx = 1:N^2; dgl_idx(blk_idx) = [];

    temp_idx = repmat(blk_idx, N, 1);
    idx = [reshape(temp_idx,  [N^2, 1]); dgl_idx'];
    jdx = [reshape(temp_idx', [N^2, 1]); dgl_idx'];

    v = zeros(length(idx), 1);

    for x = 1:length(idx)
        i = idx(x); j = jdx(x); 
        if i ~= j && D(i) ~= D(j)
            v(x) = (exp(D(i)) - exp(D(j))) / (D(i) - D(j));
        else
            v(x) = exp(D(i));
        end
    end

    Df = sparse(idx, jdx, v);
end

function Df = ddTrFunc(Df, V, H)
    Df = Df .* (V'*H*V);
    Df = V * Df * V';
end
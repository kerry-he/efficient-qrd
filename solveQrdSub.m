function [BR_feas, BR, V] = solveQrdSub(BR, V, Delta, R, kappa, opt)
    %SOLVEQRDSUB Solves the mirror descent subproblem for the quantum
    %rate-distortion problem.

    N = length(R);
    M = length(BR) / N;
    I = speye(N);

    % Compute primal iterate
    X_prev = kron(logm(PartialTrace(BR, 2, [M, N])), I);
    X = X_prev - kron(I, V) - kappa*Delta;
    [BR_U, D] = eig(X);
    BR_D = spdiag(exp(diag(D)));
    BR = BR_U * BR_D * BR_U';

    % Compute feasible primal iterate
    BR_feas = proj(BR, R);

    % Function evaluation
    obj = -trace(BR) - trace(R * V);
    grad = PartialTrace(BR, 1, [M, N]) - R;

    % Optimality gap for inexact stopping criterion
    gap = relativeEntropyQuantum(BR_feas, BR) - trace(BR_feas) + trace(BR);
    
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
            p = -Hess \ grad(:);
            p = reshape(p, [N, N]);
        else
            error(['Option sub.alg is invalid ' ...
                '(must be ''gradient'' or ''newton'')'])
        end

        t = opt.sub.t0;

        while true
            % Update dual variable
            V_new = V + t*p;
    
            % Compute primal iterate
            X = X_prev - kron(I, V_new) - kappa*Delta;
            [BR_U, D] = eig(X);
            BR_D = spdiag(exp(diag(D)));
            BR = BR_U * BR_D * BR_U';

            % Function evaluation
            obj_new = -trace(BR) - trace(R * V_new);

            % Backtracking line search
            if -obj_new <= -obj - opt.sub.alpha*t*trace(grad * p)
                break
            end
            t = t * opt.sub.beta;
        end

        % Exit if no progress is being made
        if obj - obj_new == 0
            break
        end

        grad = PartialTrace(BR, 1, [M, N]) - R;
        obj = obj_new;
        V = V_new;

        % Compute feasible primal iterate
        BR_feas = proj(BR, R);
        
        % Optimality gap for inexact stopping criterion
        gap = relativeEntropyQuantum(BR_feas, BR) - trace(BR_feas) + trace(BR);
    
        if opt.verbose == 2
            fprintf("                        |                 |  %d  \t  %.1e \n", ...
                k, gap)
        end    
        k = k + 1;

    end

end

%% Auxiliary functions

function fixed_rho = proj(rho, R)
    N = length(R);
    I = speye(N);
    fix = (R * PartialTrace(rho, 1)^-1).^0.5;
    fix = real(fix);

    fixed_rho = kron(I, fix) * rho * kron(I, fix');
end

function J = getHessian(D, U)
    N = sqrt(length(D));

    J = zeros(N^2, N^2);
    Df = getFddMatrix(D);

    for i = 1:N
        for j = 1:N
            if i > j
                temp = reshape(J((j - 1)*N + i, :), [N, N]);
                temp = temp';
                J((i - 1)*N + j, :) = temp(:);
            else
                H = sparse(i, j, 1, N, N);
                dFdH = PartialTrace(ddTrFunc(Df, U, -kron(eye(N), H)), 1, N);        
                J((i - 1)*N + j, :) = dFdH(:);
            end
        end
    end        
end

function Df = getFddMatrix(D)
    N = length(D);
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

function Df = ddTrFunc(Df, V, H)
    Df = Df .* (V'*H*V);
    Df = V * Df * V';
end
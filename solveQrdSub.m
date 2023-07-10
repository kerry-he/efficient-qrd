function [RHO, RHO_D, RHO_inf, V] = solveQrdSub(RHO, V, AR, kappa, opt)
    %SOLE_QRD_DUAL Summary of this function goes here
    %   Detailed explanation goes here

    NR = length(V);
    N = length(RHO) / NR;
    R = sparsePartialTrace(AR, 1);
    I = speye(N);

    % Cheap diagonalisation of X and V
    X_PREV = kron(spdiag(log(sparsePartialTrace(RHO, 2))), I);
    X = X_PREV - kron(I, V) + kappa*AR;
    [U, D] = sparseEig(X);

    % Function evaluation
    expD = exp(D);
    RHO = U * spdiag(expD) * U';
    obj = -trace(RHO) - sum(R .* diag(V));
    F = sparsePartialTrace(RHO, 1) - R;

    RHO_fix = fix_R(RHO, R);
    [U_fix, D_fix] = sparseEig(RHO_fix);

    gap = sparseQRE(D_fix, U_fix, expD, U) - trace(RHO_fix) + trace(RHO);
    
    fprintf("\t Error: %.5e\n", gap)

    while gap >= opt.sub.tol

        % Compute ascent direction
        if strcmp(opt.sub.alg, 'gradient')
            p = F;
        elseif strcmp(opt.sub.alg, 'newton')
            J = get_jacobian(D, U);
            p = -J \ F;
        else
            error(['Option sub.alg is invalid ' ...
                '(must be ''gradient'' or ''newton'')'])
        end

        t = opt.sub.t0;

        while true
            % Update with Newton step
            V_new = V + t*spdiag(p);
    
            % Cheap diagonalisation of X and V
            X = X_PREV - kron(I, V_new) + kappa*AR;
            [U, D] = sparseEig(X);
    
            % Function evaluation
            expD = exp(D);
            RHO = U * spdiag(expD) * U';
            obj_new = -trace(RHO) - sum(R .* diag(V_new));

            if -obj_new <= -obj - opt.sub.alpha*t*(F' * p)
                break
            end
            t = t * opt.sub.beta;
        end

        if obj - obj_new == 0
            break
        end

        F = sparsePartialTrace(RHO, 1) - R;
        obj = obj_new;
        V = V_new;

        RHO_fix = fix_R(RHO, R);
        [U_fix, D_fix] = sparseEig(RHO_fix);

        gap = sparseQRE(D_fix, U_fix, expD, U) - trace(RHO_fix) + trace(RHO);
    
        fprintf("\t Error: %.5e\n", gap)
    end

    RHO_inf = RHO;
    RHO = RHO_fix;
    RHO_D = D_fix;

end

function fixed_rho = fix_R(rho, R)
    N = length(R);
    I = speye(N);
    fix = (R ./ sparsePartialTrace(rho, 1)).^0.5;
    fix = spdiag(fix);

    fixed_rho = kron(I, fix) * rho * kron(I, fix');
end
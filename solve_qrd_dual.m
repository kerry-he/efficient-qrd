function [RHO, rho_D, V, RHO_inf] = solve_qrd_dual(RHO, V, AR, lambda, EPS)
    %SOLE_QRD_DUAL Summary of this function goes here
    %   Detailed explanation goes here

    NR = length(V);
    N = length(RHO) / NR;
    R = sparsePartialTrace(AR, 1, N);
    I = speye(N);

    % Cheap diagonalisation of X and V
    X_PREV = kron(sparse(1:N, 1:N, log(sparsePartialTrace(RHO, 2))), I);
    X = X_PREV - kron(I, V) + lambda*AR;
    [U, D] = sparseEig(X);

    % Function evaluation
    expD = exp(D);
    RHO = U * sparse(1:N^2, 1:N^2, expD) * U';
    obj = -trace(RHO) - sum(R .* diag(V));
    F = sparsePartialTrace(RHO, 1, N) - R;

    RHO_fix = fix_R(RHO, R);
    [U_fix, D_fix] = sparseEig(RHO_fix);

    gap = sparseQRE(D_fix, U_fix, expD, U) - trace(RHO_fix) + trace(RHO);
    
    fprintf("\t Error: %.5e\n", gap)

    alpha = 0.1;
    beta = 0.1;
    while gap >= EPS

        % Compute Newton step
        J = get_jacobian(D, U);
        p = J \ F;

        t = 1;
        while true
            % Update with Newton step
%             step = diag(F);
            step = -diag(p);
            V_new = V + t*step;
    
            % Cheap diagonalisation of X and V
            X = X_PREV - kron(I, V_new) + lambda*AR;
            [U, D] = sparseEig(X);
    
            % Function evaluation
            expD = exp(D);
            RHO = U * sparse(1:N^2, 1:N^2, expD) * U';
            obj_new = -trace(RHO) - sum(R .* diag(V_new));

            if -obj_new <= -obj - alpha*t*trace(diag(F) * step)
                break
            end
            t = t * beta;
        end

        if obj - obj_new == 0
            break
        end

        F = sparsePartialTrace(RHO, 1, N) - R;
        obj = obj_new;
        V = V_new;

        RHO_fix = fix_R(RHO, R);
        [U_fix, D_fix] = sparseEig(RHO_fix);

        gap = sparseQRE(D_fix, U_fix, expD, U) - trace(RHO_fix) + trace(RHO);
    
        fprintf("\t Error: %.5e\n", gap)
    end

    RHO_inf = RHO;
    RHO = RHO_fix;
    rho_D = D_fix;

end

function fixed_rho = fix_R(rho, R)
    N = length(R);
    I = speye(N);
    fix = (R ./ sparsePartialTrace(rho, 1, N)).^0.5;
    fix = sparse(1:N, 1:N, fix);

    fixed_rho = kron(I, fix) * rho * kron(I, fix');
end
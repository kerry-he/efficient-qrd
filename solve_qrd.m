function [RHO, obj] = solve_qrd(A, lambda, max_iter, tol)
    %SOLVE_QRD Summary of this function goes here
    %   Detailed explanation goes here

    arguments
        A        (:, :) double
        lambda   (1, 1) double
        max_iter {mustBeInteger} = 1000
        tol      (1, 1) double = 1e-15
    end

    N = length(A);
    
    % Pre-process input state and compute purification
    [~, A] = eig(A); A = diag(A);
    [AR, ~, AR_vec] = Purify(A);
    S_A = entropy(A);

    % Initialize primal and dual variables
    RHO = sparse(1:N^2, 1:N^2, kron(A, A));
    RHO_D = diag(RHO);
    RHO_inf = RHO;
    V = -sparse(1:N, 1:N, log(A));

    RHO_Q = sparsePartialTrace(RHO, 2, N);
    obj_prev = S_A + entropy(RHO_Q) - entropy(RHO_D) ...
        - lambda*(AR_vec' * RHO * AR_vec);
    
    EPS = 1e-2;
    
    for k = 1:max_iter

        % Perform Blahut-Arimoto iteration
        [RHO, RHO_D, V, RHO_inf] = solve_qrd_dual(RHO_inf, V, AR, lambda, EPS);
    
        % Compute objective value
        RHO_Q = sparsePartialTrace(RHO, 2);
        obj = S_A + entropy(RHO_Q) - entropy(RHO_D) ...
            - lambda*(AR_vec' * RHO * AR_vec);

        if k > 1
            EPS = max(min([abs(obj - obj_prev), EPS]), 1e-15);
        end

        fprintf("Iteration: %d \t Objective: %.5e \t EPS: %.5e\n", k, abs(obj - obj_prev), EPS)
    
        if k > 2
            if abs(obj - obj_prev) < tol
                break
            end
        end

        obj_prev = obj;
        
    end
end


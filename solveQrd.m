function [BR, obj, lb] = solveQrd(A, kappa, varargin)
    %SOLVEQRD Summary of this function goes here
    %   Detailed explanation goes here

    % Validate input arguments
    validateInputs(A, kappa);
    [opt, x0] = defaultOptionalInputs(eig(A), varargin);
    validateOptionalInputs(opt, x0);

    N = length(A);
    
    % Pre-process input state and compute purification
    A = eig(A);
    [AR, ~, AR_vec] = Purify(A);

    % Initialize primal and dual variables
    BR = x0.primal;
    BR_D = sparseEig(BR);
    BR_inf = BR;
    V = x0.dual;

    B = sparsePartialTrace(BR, 2, N);
    obj_prev = entropy(B) - entropy(BR_D) - kappa*(AR_vec' * BR * AR_vec);
    
    EPS = opt.eps_init;
    
    for k = 1:opt.max_iter

        % Perform Blahut-Arimoto iteration
        [BR, BR_D, BR_inf, V] = solve_qrd_dual(BR_inf, V, AR, kappa, EPS);
    
        % Compute objective value
        B = sparsePartialTrace(BR, 2);
        obj = entropy(B) - entropy(BR_D) - kappa*(AR_vec' * BR * AR_vec);

        if k > 1
            EPS = max(min([abs(obj - obj_prev), EPS]), opt.tol);
        end

        if opt.verbose
            fprintf("Iteration: %d \t Change in obj: %.5e \t EPS: %.5e\n", k, abs(obj - obj_prev), EPS)
        end
    
        if abs(obj - obj_prev) < opt.tol
            break
        end

        obj_prev = obj;
        
    end

    if opt.get_gap
        if opt.verbose
            fprintf("\nComputing lower bound\n")
        end

        % Solve to machine precision
        [~, ~, BR_inf, V] = solve_qrd_dual(BR_inf, V, AR, kappa, 1e-15);

        % Compute lower bound
        grad = kron((log(B) - log(sparsePartialTrace(BR_inf, 2, N))), ones(N, 1));
        grad = grad - kron(ones(N, 1), diag(V));
        lb = 0;
        for i = 1:N
            lb = lb + min(grad(i:N:end)) * A(i);
        end
    else
        lb = [];
    end

end


function [opt, x0] = defaultOptionalInputs(A, argin)

    nargin = length(argin);
    N = length(A);

    if nargin >= 0 && nargin <= 2
        if nargin == 2
            x0 = argin{2};
        else
            x0 = [];
        end

        if nargin >= 1
            opt = argin{1};
        else
            opt = [];
        end
    else
        error('Invalid number of input argumnets');
    end

    if ~isfield(opt, 'max_iter');   opt.max_iter    = 1000;     end
    if ~isfield(opt, 'tol');        opt.tol         = 1e-15;    end
    if ~isfield(opt, 'eps_init');   opt.eps_init    = 1e-2;     end
    if ~isfield(opt, 'get_gap');    opt.get_gap     = true;     end
    if ~isfield(opt, 'verbose');    opt.verbose     = 1;        end

    if ~isfield(opt, 'sub');        opt.sub         = [];       end
    if ~isfield(opt.sub, 'alg');    opt.sub.alg     = 'newton'; end
    if ~isfield(opt.sub, 't');      opt.sub.t       = 1.0;      end
    if ~isfield(opt.sub, 'alpha');  opt.sub.alpha   = 0.1;      end
    if ~isfield(opt.sub, 'beta');   opt.sub.beta    = 0.1;      end

    if ~isfield(x0, 'primal'); x0.primal = sparse(1:N^2, 1:N^2, kron(A, A)); end
    if ~isfield(x0, 'dual'); x0.dual = -sparse(1:N, 1:N, log(A)); end

end

function validateInputs(A, kappa)

    % Validate A and kappa
    if size(A, 1) ~= size(A, 2) || ~all(eig(A) >= 0, 'all')
        error('Input A must be a square PSD matrix'); 
    end
    if kappa < 0
        error('Input kappa must be nonnegative'); 
    end

end

function validateOptionalInputs(opt, x0)

    % Validate opt
    if mod(opt.max_iter, 1) ~= 0 || opt.max_iter <= 0
        error('Option max_iter must be a positive integer'); 
    end
    if opt.tol < 0
        error('Option tol must be nonnegative'); 
    end
    if opt.eps_init < 0
        error('Option eps_init must be nonnegative'); 
    end
    if ~islogical(opt.get_gap)
        error('Option get_gap must be a boolean'); 
    end
    if ~ismember(opt.verbose, [0, 1, 2])
        error('Option verbose is invalid (must be 0, 1, or 2)'); 
    end

    % Validate opt.sub
    if ~ismember(opt.sub.alg, ['gradient', 'newton'])
        error(['Option sub.alg is invalid ' ...
            '(must be ''gradient'' or ''newton'')'])
    end
    if opt.sub.t < 0
        error('Option sub.t must be nonnegative'); 
    end
    if opt.sub.alpha < 0
        error('Option sub.alpha must be nonnegative'); 
    end
    if opt.sub.beta < 0
        error('Option sub.beta must be nonnegative'); 
    end

    % Validate x0
    if size(x0.primal, 1) ~= size(x0.primal, 2) ...
            || ~all(eig(x0.primal) >= 0, 'all')
        error('Input x0.primal must be a square PSD matrix'); 
    end
    if size(x0.dual, 1) ~= size(x0.dual, 2) ...
            || ~all(x0.dual == ctranspose(x0.dual), 'all')
        error('Input x0.primal must be a square Hermitian matrix'); 
    end

end
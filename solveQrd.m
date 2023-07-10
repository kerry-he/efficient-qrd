function [rate, distortion, info] = solveQrd(A, kappa, varargin)
    %SOLVEQRD Main function to compute the quantum rate-distortion function
    %for entanglement fidelity distortion.
    %=====================================================================
    %   INPUTS
    %=====================================================================
    %   A:      Input density matrix (positive semidefinite, unit trace)
    %   kappa:  Non-negative fixed dual variable corresponding to 
    %           distortion constraint
    %
    %   opt:    (OPTIONAL) Structure containing solver options
    %   - opt.max_iter:     Maximum mirror descent iterations
    %   - opt.tol:          Tolerance at which mirror descent termiantes
    %   - opt.get_gap:      Boolean for whether to compute lower bound to
    %                       optimal value
    %   - opt.verbose:      0: Suppress all output; 1: Print outer iter.
    %                       progress; 2: Print inner and outer iter. 
    %                       progress
    %   - opt.sub:          Subproblem specific settings
    %       - opt.sub.tol:      Initial tolerance to solve to
    %       - opt.sub.alg:      'gradient': Use gradient descent; 
    %                           'newton': Use Newton's method
    %       - opt.sub.t0:       Initial step size to backtrack from
    %       - opt.sub.alpha:    Backtracking parameter
    %       - opt.sub.beta:     Backtracking parameter
    %
    %   x0:     (OPTIONAL) Structure containing initial primal and
    %           subproblem dual to initialize from
    %   - x0.primal:        Initial primal variable
    %   - x0.dual:          Intiial subproblem dual variable
    %=====================================================================
    %   OUTPUTS
    %=====================================================================
    %   rate:           Rate (nats)
    %   distortion:     Distortion
    %
    %   info:   Structure containing additional information about the solve
    %   - info.time:        Total time elapsed (s)
    %   - info.iter:        Total mirror descent (i.e., outer) iterations
    %   - info.obj_ub:      Upper bound on optimal value
    %   - info.obj_lb:      Lower bound on optimal value
    %   - info.gap:         Upper bound on optimality gap
    %   - info.primal:      Solution primal variable
    %   - info.dub_dual:    Final solution dual variable to subproblem
    %=====================================================================
    %   EXAMPLE USAGE
    %=====================================================================
    %   % Define input density matrix and dual variable with default
    %   options and intialization
    %   A = [0.2 0; 0 0.8]; kappa = 1.0; opt = []; x0 = [];
    %
    %   [rate, distortion]          = SOLVEQRD(A, kappa);
    %   [rate, distortion, info]    = SOLVEQRD(A, kappa);
    %   [rate, distortion, info]    = SOLVEQRD(A, kappa, opt);
    %   [rate, distortion, info]    = SOLVEQRD(A, kappa, opt, x0);


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

    B = sparsePartialTrace(BR, 2);
    obj_prev = entropy(B) - entropy(BR_D) - kappa*(AR_vec' * BR * AR_vec);

    timer = tic;
    
    % Main mirror descent procedure
    for k = 1:opt.max_iter

        % Perform mirror descent iteration
        [BR, BR_D, BR_inf, V] = solveQrdSub(BR_inf, V, AR, kappa, opt);
    
        % Compute objective value
        B = sparsePartialTrace(BR, 2);
        obj = entropy(B) - entropy(BR_D) - kappa*(AR_vec' * BR * AR_vec);

        % Update subproblem tolerance
        opt.sub.tol = max(min([abs(obj - obj_prev), opt.sub.tol]), opt.tol);

        if opt.verbose
            fprintf("Iteration: %d \t Change in obj: %.5e \t EPS: %.5e\n", k, abs(obj - obj_prev), opt.sub.tol)
        end
    
        % Check if we have solved to desired tolerance
        if abs(obj - obj_prev) < opt.tol
            break
        end

        obj_prev = obj;
        
    end

    % Lower bound
    if opt.get_gap
        if opt.verbose
            fprintf("\nComputing lower bound\n")
        end

        % Solve to machine precision
        opt.sub.tol = eps;
        [BR, BR_D, BR_inf, V] = solveQrdSub(BR_inf, V, AR, kappa, opt);

        % Frank-Wolfe lower bound
        grad = kron((log(B) - log(sparsePartialTrace(BR_inf, 2))), ones(N, 1));
        grad = grad - kron(ones(N, 1), diag(V));
        lb = 0;
        for i = 1:N
            lb = lb + min(grad(i:N:end)) * A(i);
        end

        B = sparsePartialTrace(BR, 2);
    else
        lb = [];
    end

    % Compute corresponding rate-distortion pair from solution
    rate = entropy(A) + entropy(B) - entropy(BR_D);
    distortion = 1 - AR_vec' * BR * AR_vec;

    % Populate info structure
    info.time = toc(timer);
    info.iter = k;
    info.obj_ub = obj + entropy(A);
    info.obj_lb = lb + entropy(A);
    if isempty(lb); info.gap = []; else; info.gap = obj - lb; end
    info.primal = BR;
    info.sub_dual = V;

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
    if ~isfield(opt, 'get_gap');    opt.get_gap     = true;     end
    if ~isfield(opt, 'verbose');    opt.verbose     = 1;        end

    if ~isfield(opt, 'sub');        opt.sub         = [];       end
    if ~isfield(opt.sub, 'tol');    opt.sub.tol     = 1e-2;     end
    if ~isfield(opt.sub, 'alg');    opt.sub.alg     = 'newton'; end
    if ~isfield(opt.sub, 't0');     opt.sub.t0      = 1.0;      end
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
    if opt.sub.tol < 0
        error('Option sub_tol must be nonnegative'); 
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
    if opt.sub.t0 < 0
        error('Option sub.t0 must be nonnegative'); 
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
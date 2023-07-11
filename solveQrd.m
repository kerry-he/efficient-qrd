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
    printHeading(N, kappa, opt.verbose)
    
    % Pre-process input state and compute purification
    A = eig(A);
    [AR, ~, AR_vec] = Purify(A);
    entrA = entropy(A);

    % Initialize primal and dual variables
    BR_feas = x0.primal; BR_D_feas = sparseEig(BR_feas);
    BR = x0.primal;
    V = x0.dual;

    % Compute rate-distortion pair and objective
    B_feas = sparsePartialTrace(BR_feas, 2);
    rate = entrA + entropy(B_feas) - entropy(BR_D_feas);
    distortion = 1 - AR_vec' * BR_feas * AR_vec;
    obj_prev = rate + kappa*distortion;

    if opt.verbose
        fprintf(" %d  \t  %.3f   %.3f | %.3f           ", ...
            0, abs(rate), abs(distortion), obj_prev)
        if opt.verbose == 1; fprintf("\n"); end
    end    

    timer = tic;
    
    % Main mirror descent procedure
    for k = 1:opt.max_iter

        % Perform mirror descent iteration
        [BR_feas, BR_D_feas, BR, V] = solveQrdSub(BR, V, AR, kappa, opt);
    
        % Compute objective value
        B_feas = sparsePartialTrace(BR_feas, 2);
        rate = entrA + entropy(B_feas) - entropy(BR_D_feas);
        distortion = 1 - AR_vec' * BR_feas * AR_vec;
        obj = rate + kappa*distortion;

        % Update subproblem tolerance
        opt.sub.tol = max(min([abs(obj - obj_prev), opt.sub.tol]), opt.tol);

        if opt.verbose
            fprintf(" %d  \t  %.3f   %.3f | %.3f   %.1e ", ...
                k, rate, distortion, obj, abs(obj - obj_prev))
            if opt.verbose == 1; fprintf("\n"); end
        end
    
        % Check if we have solved to desired tolerance
        if abs(obj - obj_prev) < opt.tol
            if opt.verbose == 2; fprintf("|\n"); end
            break
        end

        obj_prev = obj;
        
    end

    % Lower bound
    if opt.get_gap
        if opt.verbose
            fprintf("\nComputing lower bound...\n\n")
        end

        % Solve to machine precision using Newton's method
        opt.sub.tol = eps;
        opt.sub.alg = 'newton'; opt.sub.t0 = 1;
        if opt.verbose; opt.verbose = 1; end
        B = sparsePartialTrace(BR, 2);
        [~, ~, BR, V] = solveQrdSub(BR, V, AR, kappa, opt);

        % Frank-Wolfe lower bound
        grad = kron((log(B) - log(sparsePartialTrace(BR, 2))), ones(N, 1));
        grad = grad - kron(ones(N, 1), diag(V));
        lb = entrA + kappa;
        for i = 1:N
            lb = lb + min(grad(i:N:end)) * A(i);
        end
    else
        lb = [];
    end

    % Populate info structure
    info.time = toc(timer);
    info.iter = k;
    info.obj_ub = obj;
    info.obj_lb = lb;
    if ~opt.get_gap; info.gap = []; else; info.gap = obj - lb; end
    info.primal = BR_feas;
    info.sub_dual = V;

    if opt.verbose
        fprintf("Solved in %.3g seconds with rate-distortion: \n\n", info.time)
        fprintf("\tRate:       %.4f (nats)\n", rate)
        fprintf("\tDistortion: %.4f\n", distortion)
        if opt.get_gap
            fprintf("\nand solved to within optimality gap of %.1e.\n\n", info.gap)
        end
    end

end

%% Auxiliary functions

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

function printHeading(N, kappa, verbose)
    if verbose
        fprintf("Computing quantum rate-distortion with entanglement fidelity \n" + ...
            "distortion for N = %d input state and \x03BA = %.1f dist. multiplier\n\n", N, kappa);
    end

    if verbose == 2
        fprintf(" it       rate    dist  |  obj     \x0394 obj  | (sub)it  (sub)gap \n")
    elseif verbose == 1
        fprintf(" it       rate    dist  |  obj     \x0394 obj  \n")
    end
end
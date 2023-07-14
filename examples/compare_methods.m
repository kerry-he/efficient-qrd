clear all; close all; clc;

addpath(genpath('../src'))

%% Generate problem data
rng(1)
N = 8; % Density matrix size

A = randDensityMatrix(N); % Input density matrix
AR = purify(A);
kappa = 3;

fprintf("SR    Alg         Time (s)    Gap\n")

%% Solve with symmetry reduction and Newton's method
opt = [];
opt.verbose = 0;

[~, ~, info] = solveEfQrd(A, kappa, opt);
lb = info.obj_lb;
fprintf("Y     Newton      %.4f      %.1e\n", info.time, info.gap)


%% Solve with symmetry reductionand gradient descent
opt = [];
opt.verbose = 0;
opt.tol = 1e-8;
opt.sub.alg = 'gradient';
opt.sub.t0 = 1000;

[~, ~, info] = solveEfQrd(A, kappa, opt);
fprintf("Y     Gradient    %.4f      %.1e\n", info.time, info.obj_ub - lb)

%% Solve without symmetry reduction and Newton's method
opt = [];
opt.verbose = 0;

x0.primal = kron(A, eig(A, 'matrix'));
[~, ~, info] = solveQrd(A, -AR, kappa, opt, x0);
fprintf("N     Newton      %.4f      %.1e\n", info.time, info.obj - lb + kappa)

%% Solve without symmetry reduction and gradient descent
opt = [];
opt.verbose = 0;
opt.tol = 1e-8;
opt.sub.alg = 'gradient';
opt.sub.t0 = 1000;

x0.primal = kron(A, eig(A, 'matrix'));
[~, ~, info] = solveQrd(A, -AR, kappa, opt, x0);
fprintf("N     Gradient    %.4f      %.1e\n", info.time, info.obj - lb + kappa)
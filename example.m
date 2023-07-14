clear all; close all; clc;

addpath(genpath('src'))

%% Generate problem data

rng(1)
N = 4; % Density matrix size

A = randDensityMatrix(N); % Input density matrix
AR = purify(A);
kappa = 2;

opt = [];
% opt.tol = 1e-7;
% opt.sub.alg = 'gradient';
% opt.sub.t0 = 1000;
% opt.verbose = 2;

[rate, distortion, info] = solveEfQrd(A, kappa, opt);
[rate, distortion, info] = solveQrd(A, eye(N^2) - AR, kappa, opt);
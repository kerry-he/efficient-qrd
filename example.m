clear all; close all; clc;

addpath(genpath('src'))
addpath(genpath('QETLAB'))

%% Generate problem data

rng(1)
N = 8; % Density matrix size

A = RandomDensityMatrix(N, 1); % Input density matrix
[AR, R, AR_vec] = Purify(A);
kappa = 1;

opt = [];
opt.tol = 1e-10;
% opt.sub.alg = 'gradient';
% opt.sub.t0 = 1000;
opt.verbose = 2;

[rate, distortion, info] = solveQrdEF(A, kappa, opt);
[rate, distortion, info] = solveQrd(A, eye(N^2)-AR, kappa, opt);
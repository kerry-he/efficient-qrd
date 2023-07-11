clear all; close all; clc;

addpath(genpath('src'))
addpath(genpath('QETLAB'))

%% Generate problem data

rng(1)
N = 64; % Density matrix size

A = RandomDensityMatrix(N, 1); % Input density matrix
kappa = 5.0;

opt = [];
% opt.tol = 1e-15;
% opt.sub.alg = 'gradient';
% opt.sub.t0 = 1000;
% opt.verbose = 0;

[rate, distortion, info] = solveQrd(A, kappa, opt);
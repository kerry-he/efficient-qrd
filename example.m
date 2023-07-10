clear all; close all; clc;

addpath(genpath('src'))
addpath(genpath('QETLAB'))

%% Generate problem data

rng(1)
N = 32; % Density matrix size

A = RandomDensityMatrix(N, 1); % Input density matrix
kappa = 1.0;

[rate, distortion, info] = solveQrd(A, kappa);
clear all; close all; clc;

addpath(genpath('../src'))

%% Generate problem data
rng(1)
N = 64;                      % Density matrix size
A = randDensityMatrix(N);   % Input density matrix

kappa = 5;
[rate, distortion, info] = solveEfQrd(A, kappa);
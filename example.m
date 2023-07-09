clear all; close all; clc;

addpath(genpath('src'))

%% Generate problem data

rng(1)
N = 16; % Density matrix size

A = RandomDensityMatrix(N, 1); % Input density matrix
kappa = 5.0;

tic
[RHO, obj] = solveQrd(A, kappa);
toc
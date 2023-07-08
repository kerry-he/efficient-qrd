clear all; close all; clc;

addpath(genpath('src'))

%% Generate problem data

rng(1)
N = 128; % Density matrix size

A = RandomDensityMatrix(N, 1); % Input density matrix
lambda = 5.0;

tic
[RHO, obj] = solve_qrd(A, lambda);
toc
clear all; close all; clc;

addpath(genpath('../src'))

%% Generate problem data
rng(1)
N = 8;                      % Density matrix size
A = randDensityMatrix(N);   % Input density matrix

%% Compute rate-distortion curve
kappa = 12.5:-0.005:0.1;
K = length(kappa);

rate = zeros(K, 1);
distortion = zeros(K, 1);

% Specify options
opt = [];
opt.tol = 1e-8;
opt.verbose = 0;
opt.get_gap = false;
x0 = [];

% Compute rate-distortion
h = waitbar(0,'Computing quantum rate-distortion curve...');
for i = 1:K
    [rate(i), distortion(i), info] = solveEfQrd(A, kappa(i), opt, x0);

    % Warm-start using previous solution
    x0.primal = info.primal;
    x0.dual = info.sub_dual;

    waitbar(i / K)
end
close(h)

% Compute axis intercepts
rate = [2*entropyQuantum(A); rate; 0; 0];
distortion = [0; distortion; 1 - max(eig(A))^2; 1];


%% Plot curve
plot(distortion, rate)
title("Quantum rate-distortion curve")
xlabel("Distortion")
ylabel("Rate (nats)")
# Efficient Computation of the Quantum Rate-Distortion Function 

## About

A MATLAB implementation of an inexact mirror descent to compute the quantum rate-distortion function

$$\min_{\mathcal{N}\in\Phi} \quad I(\rho, \mathcal{N}) + \kappa \langle \Delta , (\mathcal{N} \otimes \mathbb{I})(|\psi\rangle\!\langle\psi|)\ \rangle$$

where $I(\rho, \mathcal{N})$ is the quantum mutual information, for an input quantum density matrix $\rho$ with purification $|\psi\rangle$ and positive distortion observable $\Delta$. In particular, for entanglement fidelity distortion $\Delta=\mathbb{I} - |\psi\rangle\!\langle\psi|$, we provide an efficient implementation of the algorithm which uses symmetry reduction to significantly reduce the dimension of dimension of the optimization problem.


## Usage

The code is completely standadlone, and should run without any additional packages. The two main functions users should use are

- `src/solveQrd.m`: Computes the quantum rate-distortion function for arbitrary distortion observable.
- `src/solveEfQrd.m`: Computes the quantum rate-distortion function specifically for entanglement fidelity distortion. Uses symmetry reduction to significantly improve computational efficiency.

An example of how to use these functions is as follows:

    A = [0.2 0; 0 0.8]; 							% Define input state
    kappa = 1.0;									% Define distortion mutliplier
    [rate, distortion, info] = solve(A, kappa); 	% Solve for rate-distortion
	
Other examples of how to use the function can be found in `examples`. These include

- `examples/entanglementFidelity.m`: Solves the entanglement fidelity rate-distortion for a random input density matrix.
- `examples/computeCurve.m`: Traces out the entire entanglement fidelity rate-distortion curve for a random input density matrix.
- `examples/compareMethods.m`: Demonstration of various possible methods to solve for the entanglement fidelity rate-distortion.


## Citation

TODO
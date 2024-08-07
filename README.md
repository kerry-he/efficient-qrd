# Efficient Computation of the Quantum Rate-Distortion Function 

## About

A MATLAB implementation of an inexact mirror descent algorithm to compute the quantum rate-distortion function for an input quantum density matrix $\rho$ with purification $|\psi\rangle$, positive distortion observable $\Delta$, and distortion multiplier $\kappa\geq0$

$$\min_{\mathcal{N}} \quad I(\rho, \mathcal{N}) + \kappa \langle \Delta , (\mathcal{N} \otimes \text{Id.})(|\psi\rangle\langle\psi|) \rangle, \qquad \text{subj. to} \quad \mathcal{N}\in\mathsf{CPTP}$$

where $I(\rho, \mathcal{N})$ is the quantum mutual information, $\mathsf{CPTP}$ is the set of completely positive trace preserving quantum channels. In particular, for entanglement fidelity distortion $\Delta=\text{Id.} - |\psi\rangle\langle\psi|$, we provide an efficient implementation of the algorithm which uses symmetry reduction to significantly reduce the dimension of dimension of the optimization problem.


## Usage

The code is completely standadlone, and should run without any additional packages. The two main functions users should use are

- `src/solveQrd.m`: Computes the quantum rate-distortion function for arbitrary distortion observable.
- `src/solveEfQrd.m`: Computes the quantum rate-distortion function specifically for entanglement fidelity distortion. Uses symmetry reduction to significantly improve computational efficiency.

An example of how to use these functions is as follows:

	A = [0.2 0; 0 0.8]; 					% Define input state
	kappa = 1.0;						% Define distortion mutliplier
	[rate, distortion, info] = solveEfQrd(A, kappa); 	% Solve for rate-distortion
	
Other examples of how to use the function can be found in `examples`. These include

- `examples/entanglementFidelity.m`: Solves the entanglement fidelity rate-distortion for a random input density matrix.
- `examples/computeCurve.m`: Traces out the entire entanglement fidelity rate-distortion curve for a random input density matrix.
- `examples/compareMethods.m`: Demonstration of various possible methods to solve for the entanglement fidelity rate-distortion.


## Citation

This code is based on the work the paper here: <https://quantum-journal.org/papers/q-2024-04-09-1314/>. If you find our work useful, please cite us using

	@article{he2024efficient,
	  title={Efficient Computation of the Quantum Rate-Distortion Function},
	  author={He, Kerry and Saunderson, James and Fawzi, Hamza},
	  journal={Quantum},
	  volume={8},
	  pages={1314},
	  year={2024},
	  publisher={Verein zur F{\"o}rderung des Open Access Publizierens in den Quantenwissenschaften},
	  doi = {10.22331/q-2024-04-09-1314},
	}

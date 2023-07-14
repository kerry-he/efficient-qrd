function S = relativeEntropyQuantum(X, Y)
    %RELATIVEENTROPYQUANTUM Copmutes the quantum relative entropy between
    %two positive definite matrices

    S = trace(X * (logm(X) - logm(Y)));
end


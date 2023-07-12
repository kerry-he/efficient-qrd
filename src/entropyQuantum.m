function S = entropyQuantum(X)
    %ENTROPYQUANTUM Computes the quantum relative entropy of a positive
    %definite matrix

    S = -trace(X * logm(X));
end


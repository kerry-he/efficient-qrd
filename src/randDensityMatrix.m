function X = randDensityMatrix(N)
    %RANDDENSITYMATRIX Generates a random unit trace, positive semidefinite
    %matrix of dimension N.

    X = rand(N);
    X = X * X';
    X = X / trace(X);
end


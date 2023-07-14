function varargout = sparseEig(X)
    %SPARSEEIG Eigenvalues and eigenvectors for matrices in fixed-point
    %subspace.
    %   D = SPARSEEIG(X) produces a column vector D containing the
    %   eigenvalues of a square matrix X.
    %
    %   [U, D] = SPARSEEIG(X) produces a column vector D of eigenvalues and
    %   a unitary matrix U whose columns are the eigenvectors of a square
    %   matrix X.

    % Check arguments
    nargoutchk(1, 2)

    N = sqrt(length(X));

    % Create sparse indexing
    blk_idx = 1:N+1:N^2;
    dgl_idx = 1:N^2; dgl_idx(blk_idx) = [];

    blk = full(X(blk_idx, blk_idx));
    D = full(diag(X));

    if nargout == 1
        
        % Only return eigenvalues
        D_blk = eig(blk);
        D(blk_idx) = D_blk;

        varargout = {D};

    elseif nargout == 2

        % Return eigenvectors and eigenvalues
        [U_blk, D_blk] = eig(blk, 'vector');
        D(blk_idx) = D_blk;
    
        temp_idx = repmat(blk_idx, N, 1);
        idx = [reshape(temp_idx,  [N^2, 1]); dgl_idx'];
        jdx = [reshape(temp_idx', [N^2, 1]); dgl_idx'];
        v   = [reshape(U_blk',    [N^2, 1]); ones(N^2-N, 1)];
        U = sparse(idx, jdx, v);

        varargout = {U, D};

    end
end
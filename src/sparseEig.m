function [U, D] = sparseEig(X)
    %SPARSEEIG Summary of this function goes here
    %   Detailed explanation goes here
    
    N = sqrt(length(X));

    blk_idx = 1:N+1:N^2;
    dgl_idx = 1:N^2; dgl_idx(blk_idx) = [];

    blk = full(X(blk_idx, blk_idx));

    [U_blk, D_blk] = eig(blk);

    D = full(diag(X));
    D(blk_idx) = D_blk(blk_idx);

    temp_idx = repmat(blk_idx, N, 1);
    idx = [reshape(temp_idx,  [N^2, 1]); dgl_idx'];
    jdx = [reshape(temp_idx', [N^2, 1]); dgl_idx'];
    v   = [reshape(U_blk',    [N^2, 1]); ones(N^2-N, 1)];
    U = sparse(idx, jdx, v);
end


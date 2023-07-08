function S = sparseQRE(rho_D, rho_U, sigma_D, sigma_U)

    N = sqrt(length(rho_D));

    blk_idx = 1:N+1:N^2;
    dgl_idx = 1:N^2; dgl_idx(blk_idx) = [];

    % Compute diagonal part of entropy
    S = relative_entropy(rho_D(dgl_idx), sigma_D(dgl_idx));

    % Compute block part of entropy
    rho_D_blk = full(rho_D(blk_idx));
    rho_U_blk = full(rho_U(blk_idx, blk_idx));
    sigma_D_blk = full(sigma_D(blk_idx));
    sigma_U_blk = full(sigma_U(blk_idx, blk_idx));

    rho_sigma_U = rho_U_blk' * sigma_U_blk;

    S = S - entropy(rho_D_blk);
    S = S - trace(sparse(1:N, 1:N, rho_D_blk) * rho_sigma_U * sparse(1:N, 1:N, log(sigma_D_blk)) * rho_sigma_U');
end

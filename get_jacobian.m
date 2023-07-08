function J = get_jacobian(D, U)
    N = sqrt(length(D));

    J = zeros(N, N);
    Df = exp_fdd(D);

    for i = 1:N
        H = sparse(i:N:N^2, i:N:N^2, -1, N^2, N^2);
        J_temp = fdd(Df, U, H);
        J(i, :) = sparsePartialTrace(J_temp, 1, N);
    end
end


function Df = exp_fdd(D)

    N = sqrt(length(D));

    blk_idx = 1:N+1:N^2;
    dgl_idx = 1:N^2; dgl_idx(blk_idx) = [];

    temp_idx = repmat(blk_idx, N, 1);
    idx = [reshape(temp_idx,  [N^2, 1]); dgl_idx'];
    jdx = [reshape(temp_idx', [N^2, 1]); dgl_idx'];

    v = zeros(length(idx), 1);

    for x = 1:length(idx)
        i = idx(x); j = jdx(x); 
        if i ~= j && D(i) ~= D(j)
            v(x) = (exp(D(i)) - exp(D(j))) / (D(i) - D(j));
        else
            v(x) = exp(D(i));
        end
    end

    Df = sparse(idx, jdx, v);
end


function Df = fdd(Df, V, H)
    Df = Df .* (V'*H*V);
    Df = V * Df * V';
end
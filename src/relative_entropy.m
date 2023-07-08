function H = relative_entropy(p, q)
    H = sum(p .* log(p ./ q));
end
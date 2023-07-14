function H = relativeEntropy(p, q)
    H = sum(p .* log(p ./ q));
end
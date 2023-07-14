function H = entropy(x)
    %ENTROPY Computes the Shannon entropy of a positive vector

    H = -sum(x .* log(x), 'all');
end


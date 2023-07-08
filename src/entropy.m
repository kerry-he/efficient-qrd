function H = entropy(x)
    %ENTROPY Summary of this function goes here
    %   Detailed explanation goes here
    H =  -sum(x .* log(x), 'all');
end


clear all; close all; clc;

% Define function that you want to check
m = 5; n = 4;
p = rand(m, 1); p = p / sum(p);


M = 1000000;

for j = 1:M

    % Pick two points randomly in the domain
%     a = rand(m, 1); a = a / sum(a);
%     b = rand(m, 1); b = b / sum(b);
    a = rand(n, m); a = a ./ sum(a);
    b = rand(n, m); b = b ./ sum(b);
    
    % Generate line between randomly chosen points
    N = 100;
    ft = zeros(1, N);
    theta = linspace(0., 1., N);
    
    for i = 1:N
        x = theta(i) * a + (1 - theta(i)) * b;
        ft(i) = I(p, x);
    end
    
    % Automatically check to make sure second derivative is positive
    dfdt = ft(2:end) - ft(1:end-1);
    d2fdt2 = dfdt(2:end) - dfdt(1:end-1);
    
    if sum(d2fdt2 <= 0)
        fprintf("Not convex!\n")
        plot(theta, ft)
    end

end



% Define function that you want to check
function out = I(p, Q)
    q = Q*p;
    [N, M] = size(Q);

    out = 0;

    for j = 1:M
        for i = 1:N
            if Q(i, j) ~= 0
                out = out + p(j) * Q(i, j) * log(Q(i, j) / q(i));
            end
        end
    end
end

function out = relent(p)
    out = 0;
    for i = 1:length(p)

        if p(i) ~= 0
            out = out + p(i) * log(p(i));
        end
    end
end
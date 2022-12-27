classdef QRateDistortion < handle
    %QRATEDISTORTION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        A           % Input density matrix
        AR          % Purified input density matrix
        AR_vec      % Purified input state vector
        N           % Size of input density matrix
        NR          % Size of purified component
        NAR         % Size of combined purified component
        
        I           % Sparse block I matrix for Kroneker and partial trace 
        J           % Sparse block J matrix for Kroneker and partial trace

        I_K         % Sparse linear operator for (N o I)(phi_QR)
        I_adj_K         

        i_K         % Sparse linear operator for N(phi_Q)
        i_adj_K

        F           % Fidelity linear operator for Fe(N)
        F_grad

        H           % Directional derivatives      
        B           % BFGS Hessian approximation
    end
    
    methods
        function obj = QRateDistortion(A)
            % Constructor
            % Quantum input states
            obj.A = A;
            obj.AR_vec = purification(A); 
            obj.AR = obj.AR_vec * obj.AR_vec';
            obj.N = length(A);
            obj.NR = length(obj.AR) / obj.N;
            obj.NAR = length(obj.AR);
            
            % Useful sparse operators
            obj.I = cell(obj.N, 1);
            obj.J = cell(obj.N, 1);
            
            for i = 1:obj.N
                obj.I{i} = sparse((1:obj.N)+(i-1)*obj.N, 1:obj.N, ...
                                  ones(obj.N, 1), obj.N^2, obj.N);
                obj.J{i} = sparse(i:obj.N:obj.N^2, 1:obj.N, ...
                                  ones(obj.N, 1), obj.N^2, obj.N);
            end

            % Precompute linear operators
            tic
            [obj.I_K, obj.I_adj_K] = obj.precompute_I_map();
            [obj.i_K, obj.i_adj_K] = obj.precompute_i_map();
            [obj.F, obj.F_grad] = obj.precompute_F_map();

            obj.H = cell(obj.N, obj.N);
            for i = 1:obj.N
                for j = 1:obj.N
                    H = zeros(obj.N);
                    H(i, j) = 1;
                    obj.H{i, j} = -obj.I_adj_inv_map(kron(H, eye(obj.N)));            
                end
            end
            obj.B = [];

            fprintf(1, 'Finished precomputation of linear operators in %.2f sec\n', toc);
        end
        
        function func = q_mutual_inf(obj, PHI)
            % Objective function
            func = von_neumann_entropy(obj.A);
            func = func + von_neumann_entropy(obj.i_map(PHI));
            func = func - von_neumann_entropy(obj.I_map(PHI));
        end

        function grad = q_mutual_inf_grad(obj, PHI)
            % Gradient of objective function
            grad = obj.I_adj_map( logm(obj.I_map(PHI)) + eye(obj.NAR) );
            grad = grad - obj.i_adj_map( logm(obj.i_map(PHI)) + eye(obj.N) );
        end

%         function V = trace_preserve_V(obj, X, V)
%             % Trace preserving operation using Newton root finding
%             while true
%                 VoI = obj.I_adj_inv_map(kron(V, eye(obj.N)));
% 
%                 F = PartialTrace(obj.I_inv_map( expm(X - VoI) ), 2) - eye(obj.N);
%                 F = F(:);
%         
%                 if norm(F) < 1e-10
%                     break
%                 end
%             
%                 dF = zeros(obj.N^2, obj.N^2);
%                 
%                 % Compute Jacobian
%                 for i = 1:obj.N
%                     for j = 1:obj.N
%                         dFdH = PartialTrace(obj.I_inv_map(exp_fdd(X - VoI, ...
%                             obj.H{i, j})), 2);                                        
%                         dF((i - 1)*obj.N + j, :) = dFdH(:);
%                     end
%                 end
%             
%                 % Line search
%                 alpha = 1.0;
%                 beta = 0.9;
%                 while true
%                     V_vec = V(:) - alpha*(dF \ F);
%                     V_new = reshape(V_vec, size(V));
%                     V_new = (V_new + V_new') / 2;
% 
%                     VoI = obj.I_adj_inv_map(kron(V_new, eye(obj.N)));
% 
%                     F_new = PartialTrace(obj.I_inv_map( expm(X - VoI) ), 2) - eye(obj.N);
%                     
%                     if norm(F_new) <= norm(F)
%                         break
%                     end
%                     
%                     alpha = alpha * beta;
%                 end
%                 
%                 V = V_new;
%                 
% %                 break
%             end
%         end
        function V = trace_preserve_V(obj, X, V, bfgs)
            % Trace preserving operation using Newton root finding
            while true
%                 tic 
                VoI = obj.I_adj_inv_map(kron(V, eye(obj.N)));

                F = PartialTrace(obj.I_inv_map( expm(X - VoI) ), 2, obj.N) - eye(obj.N);
                F = F(:);
%                 fprintf("Computed function value: %.2f\n", toc)
        
                if norm(F) < 1e-10
                    break
                end
            
                dF = zeros(obj.N^2, obj.N^2);
                
                % Compute Jacobian
                if isempty(obj.B)
%                     tic
                    for i = 1:obj.N
                        for j = 1:obj.N
                            dFdH = PartialTrace(obj.I_inv_map(exp_fdd(X - VoI, ...
                                obj.H{i, j})), 2, obj.N);                                        
                            dF((i - 1)*obj.N + j, :) = dFdH(:);
                        end
                    end
                    obj.B = inv(dF);
%                     fprintf("Computed initial Jacobian: %.2f\n", toc)
                end

                % Compute step direction
                p = -obj.B * F;
            
%                 tic
                % Line search
                alpha = 1.0;
                beta = 0.1;
                for i = 1:3
                    s = alpha * p;
                    V_vec = V(:) + s;
                    V_new = reshape(V_vec, size(V));
%                     V_new = (V_new + V_new') / 2;

                    VoI = obj.I_adj_inv_map(kron(V_new, eye(obj.N)));

                    F_new = PartialTrace(obj.I_inv_map( expm(X - VoI) ), 2) - eye(obj.N);
                    F_new = F_new(:);
                    
                    if norm(F_new) <= norm(F)
                        break
                    end
%                     break
                    alpha = alpha * beta;
                end
                
                V = V_new;

%                 fprintf("Solved for step size: %.2f\n \t Error: %.2e\n", toc, norm(F_new))
                
                y = F_new - F;
                
                if ~bfgs
                    obj.B = [];
                    fprintf("\t Error: %.5e\n", norm(F_new))
                else
    %                 obj.B = obj.B + (s'*y + y'*obj.B*y)*(s*s')/(s'*y)^2 - (obj.B*y*s' + s*y'*obj.B)/(s'*y);
    %                 obj.B = obj.B - (obj.B*(y*y')*obj.B) / (y'*obj.B*y) + (s*s')/(s'*y);
                    rho = 1 / (y'*s);
                    obj.B = (eye(obj.N^2) - rho*s*y') * obj.B * (eye(obj.N^2) - rho*y*s') + rho*(s*s');
    %                 obj.B = obj.B + (s - obj.B*y)/(s'*obj.B*y) * s'*obj.B;
                end                

                
%                 break
            end
        end
        
        function [I_K, I_adj_K] = precompute_I_map(obj)
            KI = zeros(obj.N^6, 1);
            KJ = zeros(obj.N^6, 1);
            KV = zeros(obj.N^6, 1);

            KI_adj = zeros(obj.N^6, 1);
            KJ_adj = zeros(obj.N^6, 1);
            KV_adj = zeros(obj.N^6, 1);

            K_size = 0;

            for i = 1:obj.N
            for j = 1:obj.N      
                for x = 1:obj.N
                for y = 1:obj.N
                        % Compute normal I operator
                        AR_ijxy = obj.AR((i - 1)*obj.N + x, (j - 1)*obj.N + y);
                        [I, J, V] = find( ...
                            kron(obj.J{x}*obj.I{i}', obj.J{y}*obj.I{j}') * AR_ijxy);

                        size = length(V);

                        KI(K_size + 1 : K_size + size) = I;
                        KJ(K_size + 1 : K_size + size) = J;
                        KV(K_size + 1 : K_size + size) = V;

                        % Compute adjoint I operator
                        [I, J, V] = find( ...
                            kron(obj.I{j}*obj.J{y}', obj.I{i}*obj.J{x}') * AR_ijxy);

                        size = length(V);

                        KI_adj(K_size + 1 : K_size + size) = I;
                        KJ_adj(K_size + 1 : K_size + size) = J;
                        KV_adj(K_size + 1 : K_size + size) = V;                   

                        K_size = K_size + size;
                end
                end
            end 
            end
            KI = KI(1:K_size); KJ = KJ(1:K_size); KV = KV(1:K_size);
            KI_adj = KI_adj(1:K_size); KJ_adj = KJ_adj(1:K_size); KV_adj = KV_adj(1:K_size);

            I_K = sparse(KI, KJ, KV, obj.N^4, obj.N^4);
            I_adj_K = sparse(KI_adj, KJ_adj, KV_adj, obj.N^4, obj.N^4);
        end

%         function [I_K, I_adj_K] = precompute_I_map(obj)
%             KI = zeros(obj.N^6, 1);
%             KJ = zeros(obj.N^6, 1);
%             KV = zeros(obj.N^6, 1);
% 
%             KI_adj = zeros(obj.N^6, 1);
%             KJ_adj = zeros(obj.N^6, 1);
%             KV_adj = zeros(obj.N^6, 1);
% 
%             K_size = 0;
% 
%             II = cell(obj.NAR, 1);
%             JJ = cell(obj.NAR, 1);
%             
%             for i = 1:obj.NAR
%                 II{i} = sparse((1:obj.NAR)+(i-1)*obj.NAR, 1:obj.NAR, ...
%                         ones(obj.NAR, 1), obj.NAR^2, obj.NAR);
%             end
%             for i = 1:obj.NR
%                 JJ{i} = sparse(i:obj.NR:obj.NAR, 1:obj.N, ...
%                        ones(obj.N, 1), obj.NAR, obj.N);
%             end
% 
%             
% 
%             for i = 1:obj.NR
%             for j = 1:obj.NR     
%                 for x = 1:obj.NAR
%                 for y = 1:obj.NAR
%                         % Compute normal I operator
%                         AR_xy = obj.AR(x, y);
%                         [I, J, V] = find( ...
%                             kron(II{x}'*kron(JJ{i}, JJ{i}), II{y}'*kron(JJ{j}, JJ{j})) * AR_xy);
% 
%                         size = length(V);
% 
%                         KI(K_size + 1 : K_size + size) = I;
%                         KJ(K_size + 1 : K_size + size) = J;
%                         KV(K_size + 1 : K_size + size) = V;
% 
%                         % Compute adjoint I operator
%                         [I, J, V] = find( ...
%                             kron(kron(JJ{j}, JJ{j})'*II{y}, kron(JJ{i}, JJ{i})'*II{x}) * AR_xy);
% 
%                         size = length(V);
% 
%                         KI_adj(K_size + 1 : K_size + size) = I;
%                         KJ_adj(K_size + 1 : K_size + size) = J;
%                         KV_adj(K_size + 1 : K_size + size) = V;                   
% 
%                         K_size = K_size + size;
%                 end
%                 end
%             end 
%             end
%             KI = KI(1:K_size); KJ = KJ(1:K_size); KV = KV(1:K_size);
%             KI_adj = KI_adj(1:K_size); KJ_adj = KJ_adj(1:K_size); KV_adj = KV_adj(1:K_size);
% 
%             I_K = sparse(KI, KJ, KV, obj.NAR^2, obj.N^4);
%             I_adj_K = sparse(KI_adj, KJ_adj, KV_adj, obj.N^4, obj.NAR^2);
%         end

        function [i_K, i_adj_K] = precompute_i_map(obj)
            KI = zeros(obj.N^4, 1);
            KJ = zeros(obj.N^4, 1);
            KV = zeros(obj.N^4, 1);

            KI_adj = zeros(obj.N^4, 1);
            KJ_adj = zeros(obj.N^4, 1);
            KV_adj = zeros(obj.N^4, 1);

            K_size = 0;

            for i = 1:obj.N
                for j = 1:obj.N
                    % Compute normal i operator
                    [I, J, V] = find( ...
                        kron(obj.I{i}', obj.I{j}') * obj.A(i, j));

                    size = length(V);

                    KI(K_size + 1 : K_size + size) = I;
                    KJ(K_size + 1 : K_size + size) = J;
                    KV(K_size + 1 : K_size + size) = V;

                    % Compute adjoint i operator
                    [I, J, V] = find( ...
                        kron(obj.I{j}, obj.I{i}) * obj.A(i, j));

                    KI_adj(K_size + 1 : K_size + size) = I;
                    KJ_adj(K_size + 1 : K_size + size) = J;
                    KV_adj(K_size + 1 : K_size + size) = V;

                    K_size = K_size + size;                  
                end 
            end
            KI = KI(1:K_size); KJ = KJ(1:K_size); KV = KV(1:K_size);
            KI_adj = KI_adj(1:K_size); KJ_adj = KJ_adj(1:K_size); KV_adj = KV_adj(1:K_size);
            
            i_K = sparse(KI, KJ, KV, obj.N^2, obj.N^4);
            i_adj_K = sparse(KI_adj, KJ_adj, KV_adj, obj.N^4, obj.N^2);
        end

        function [F, F_grad] = precompute_F_map(obj)
            F = zeros(1, obj.N^4);
            F_grad = zeros(obj.N^2, obj.N^2);
            for i = 1:obj.N
            for j = 1:obj.N
                for x = 1:obj.N
                for y = 1:obj.N
                    AR_ijxy = obj.AR((i - 1)*obj.N + x, (j - 1)*obj.N + y);
                    F = F + kron(obj.AR_vec' * obj.J{x}*obj.I{i}', ...
                        obj.AR_vec' * obj.J{y}*obj.I{j}') * AR_ijxy;

                    F_grad = F_grad + (obj.AR_vec' * obj.J{x} * obj.I{i}')' ...
                        * (obj.I{j} * obj.J{y}' * obj.AR_vec)' * AR_ijxy;
                end
                end
            end 
            end
        end

%         function [F, F_grad] = precompute_F_map(obj)
%             F = zeros(1, obj.N^4);
%             F_grad = zeros(obj.N^2, obj.N^2);
% 
%             II = cell(obj.NAR, 1);
%             JJ = cell(obj.NAR, 1);
%             
%             for i = 1:obj.NAR
%                 II{i} = sparse((1:obj.NAR)+(i-1)*obj.NAR, 1:obj.NAR, ...
%                         ones(obj.NAR, 1), obj.NAR^2, obj.NAR);
%             end
%             for i = 1:obj.NR
%                 JJ{i} = sparse(i:obj.NR:obj.NAR, 1:obj.N, ...
%                        ones(obj.N, 1), obj.NAR, obj.N);
%             end
% 
%             for i = 1:obj.NR
%             for j = 1:obj.NR     
%                 for x = 1:obj.NAR
%                 for y = 1:obj.NAR
%                     % Compute normal I operator
%                     AR_xy = obj.AR(x, y);
% 
%                     F = F + kron(obj.AR_vec' * II{x}'*kron(JJ{i}, JJ{i}), ...
%                         obj.AR_vec' * II{y}'*kron(JJ{j}, JJ{j})) * AR_xy;
% 
%                     F_grad = F_grad + (obj.AR_vec' * II{x}'*kron(JJ{i}, JJ{i}))' ...
%                         * (kron(JJ{j}, JJ{j})'*II{y} * obj.AR_vec)' * AR_xy;
%                 end
%                 end
%             end 
%             end
%         end
        
        function out = I_map(obj, PHI)
            out = obj.I_K * PHI(:);
            out = reshape(out, [obj.NAR, obj.NAR]);
        end

        function out = I_adj_map(obj, PHI)
            out = obj.I_adj_K * PHI(:);
            out = reshape(out, [obj.N^2, obj.N^2]);
        end

        function out = I_inv_map(obj, PHI)
            out = obj.I_K \ PHI(:);
%             out = pinv(full(obj.I_K)) * PHI(:);
            out = reshape(out, [obj.N^2, obj.N^2]);
        end

        function out = I_adj_inv_map(obj, PHI)
            out = obj.I_adj_K \ PHI(:);
%             out = pinv(full(obj.I_adj_K)) * PHI(:);
            out = reshape(out, [obj.NAR, obj.NAR]);
        end        
        
        function out = i_map(obj, PHI)
            out = obj.i_K * PHI(:);
            out = reshape(out, [obj.N, obj.N]);
        end

        function out = i_adj_map(obj, PHI)
            out = obj.i_adj_K * PHI(:);
            out = reshape(out, [obj.N^2, obj.N^2]);
        end

        function out = fidelity(obj, PHI)
            out = obj.F * PHI(:);
        end        
        
    end
end

%% Other helper functions
function S = von_neumann_entropy(rho)
    rho = rho + eye(size(rho)) * 1e-10;
    S = -trace(rho * logm(rho));
end

function Df = exp_fdd(A, H)
    [V, D] = eig(A);
    N = length(A);

    Df = zeros(N, N);

    for i = 1:N
        for j = 1:N
            if i == j || D(i, i) == D(j, j)
                Df(i, j) = exp(D(i, j));
            else
                Df(i, j) = (exp(D(i, i)) - exp(D(j, j))) / (D(i, i) - D(j, j));
            end
        end
    end
    Df = Df .* (V'*H*V);
    Df = V * Df * V';
end
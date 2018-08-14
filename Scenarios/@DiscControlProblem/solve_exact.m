function [ controller, obj_val ] = solve_exact(Obj)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    horizon = Obj.Parameters.Horizon;
    [n, ~, m] = size(Obj.Transitions);
    
    if horizon == Inf
        Obj.Controller.Policy = zeros(m, n);
        
        % Compute value function exactly
        error('Not Implemented');
        
        % Compute policy from value function
        for i = 1:n
            idx = 1;
            val = Inf;
            
            for j = 1:m
                if Obj.Costs(i, j) + Obj.Transitions(:, i, j)' * V < val
                    idx = j;
                    val = Obj.Costs(i, j) + Obj.Transitions(:, i, j)' * V;
                end
            end
            
            Obj.Controller(idx, i) = 1;
        end
        
        controller = Obj.Controller.Policy;
        obj_val = cvx_optval;
    else
        V = [zeros(n, horizon) Obj.TerminalCosts];
        Obj.Controller.Policy = zeros(m, n, horizon);
        
        for t = horizon:-1:1
            for i = 1:n
                idx = 1;
                val = Inf;

                for j = 1:m
                    if Obj.Costs(i, j) + Obj.Transitions(:, i, j)' * V(:, t + 1) < val
                        idx = j;
                        val = Obj.Costs(i, j) + Obj.Transitions(:, i, j)' * V(:, t + 1);
                    end
                end

                Obj.Controller.Policy(idx, i, t) = 1;
            end
            
            for i = 1:n
                V(i, t) = Obj.Costs(i, :) * Obj.Controller.Policy(:, i, t) + V(:, t + 1)' * (squeeze(Obj.Transitions(:, i, :)) * Obj.Controller.Policy(:, i, t));
            end
        end
        
        obj_val = V(:, 1)' * Obj.Init;
    end
    
    Obj.SolverName = 'Exact';
    controller = Obj.Controller;
end


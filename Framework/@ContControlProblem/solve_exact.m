function [controller, obj_val, obj_hist] = solve_exact(Obj)

if isinf(Obj.Parameters.Horizon)
    [controller, obj_val, obj_hist] = solve_exact_inf(Obj);
else
    [controller, obj_val, obj_hist] = solve_exact_horizon(Obj, Obj.Parameters.Horizon);
end

Obj.Controller = controller;
Obj.SolverName = 'Exact';

end

function [controller, obj_val, obj_hist] = solve_exact_inf(Obj)
    n = Obj.NStates;
    m = Obj.NInputs;
    
    [A, B] = linearize(Obj, zeros(n, 1), zeros(m, 1));
    [Q, R] = costs(Obj, zeros(n, 1));
    
    [controller, P, ~] = dlqr(A, B, Q, R);
    
    controller = -controller;
    obj_val = Obj.Init' * P * Obj.Init;
    obj_hist = obj_val;
end

function [best_controller, obj_val, obj_hist] = solve_exact_horizon(Obj, horizon)
    n = Obj.NStates;
    m = Obj.NInputs;
    
    controller = zeros(m, n, horizon);
    
    obj_hist = zeros(Obj.SolverOptions.Iters + 1, 1);
    obj_val = inf;
    
    for iter = 1:Obj.SolverOptions.Iters
        states = [Obj.Init zeros(n, horizon)];
        inputs = zeros(m, horizon);
        
        % Forward equations
        for t = 1:horizon
            inputs(:, t) = controller(:, :, t) * states(:, t);
            states(:, t + 1) = dynamics(Obj, states(:, t), inputs(:, t));
            obj_hist(iter) = obj_hist(iter) + cost(Obj, states(:, t), inputs(:, t));
        end
        
        obj_hist(iter) = obj_hist(iter) + terminal_cost(Obj, states(:, end));
        
        if obj_hist(iter) < obj_val
            best_controller = controller;
            obj_val = obj_hist(iter);
        end
        
        % Backward equations
        P = quadraticize_terminal_cost(Obj, states(:, end));
        
        for t = horizon:-1:1
            [A, B] = linearize(Obj, states(:, t), inputs(:, t));
            [Q, R] = quadraticize_cost(Obj, states(:, t), inputs(:, t)); 
            
            P = Q + A' * P * A - A' * P * B * inv(R + B' * P * B) * B' * P * A;
            controller(:, :, t) = -inv(R + B' * P * B) * B' * P * A;
        end
    end
    
    % Final forward pass
    for t = 1:horizon
        inputs(:, t) = controller(:, :, t) * states(:, t);
        states(:, t + 1) = dynamics(Obj, states(:, t), inputs(:, t));
        obj_hist(end) = obj_hist(end) + cost(Obj, states(:, t), inputs(:, t));
    end
        
    obj_hist(end) = obj_hist(end) + terminal_cost(Obj, states(:, end));
    
    if obj_hist(end) < obj_val
        best_controller = controller;
        obj_val = obj_hist(end);
    end
end
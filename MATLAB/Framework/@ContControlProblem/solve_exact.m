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
    
    K = zeros(m, n, horizon);
    f = zeros(m, horizon);
    
    P = zeros(n, n, horizon + 1);
    b = zeros(n, horizon + 1);
    g = Obj.Parameters.Goals;
    
    states = [Obj.Init.mean zeros(n, horizon)];
    inputs = zeros(m, horizon);
    
    obj_hist = zeros(Obj.SolverOptions.Iters + 1, 1);
    obj_val = inf;
    
    for iter = 1:Obj.SolverOptions.Iters        
        
        % Forward equations
        for t = 1:horizon
            inputs(:, t) = K(:, :, t) * states(:, t) + f(:, t);
            states(:, t + 1) = dynamics(Obj, states(:, t), inputs(:, t));
            obj_hist(iter) = obj_hist(iter) + cost(Obj, states(:, t), inputs(:, t), t);
        end
        
        obj_hist(iter) = obj_hist(iter) + terminal_cost(Obj, states(:, end));
        
        if obj_hist(iter) < obj_val
            best_controller.K = K;
            best_controller.f = f;
            best_controller.nominal_inputs = zeros(Obj.Parameters.NInputs, Obj.Parameters.Horizon);
            best_controller.nominal_states = zeros(Obj.Parameters.NStates, Obj.Parameters.Horizon + 1);
            
            obj_val = obj_hist(iter);
        end
        
        % Backward equations
        P(:, :, end) = quadraticize_terminal_cost(Obj, states(:, end));
        b(:, end) = -P(:, :, end) * g(:, end);
        
        for t = horizon:-1:1
            [A, B] = linearize(Obj, states(:, t), inputs(:, t));
            [Q, R] = quadraticize_cost(Obj, states(:, t), inputs(:, t)); 
            
            K(:, :, t) = -inv(R + B' * P(:, :, t + 1) * B) * B' * P(:, :, t + 1) * A;
            f(:, t) = -inv(R + B' * P(:, :, t + 1) * B) * B' * b(:, t + 1);
            
            P(:, :, t) = Q + A' * P(:, :, t + 1) * A - A' * P(:, :, t + 1) * B * inv(R + B' * P(:, :, t + 1) * B) * B' * P(:, :, t + 1) * A;
            b(:, t) = -Q * g(:, t) + K(:, :, t)' * R * f(:, t) + ...
                (A + B * K(:, :, t))' * P(:, :, t) * B * f(:, t) + (A + B * K(:, :, t))' * b(:, t + 1);
        end
    end
    
    % Final forward pass
    for t = 1:horizon
        inputs(:, t) = K(:, :, t) * states(:, t) + f(:, t);
        states(:, t + 1) = dynamics(Obj, states(:, t), inputs(:, t));
        obj_hist(iter + 1) = obj_hist(iter + 1) + cost(Obj, states(:, t), inputs(:, t), t);
    end
        
    obj_hist(iter + 1) = obj_hist(iter + 1) + terminal_cost(Obj, states(:, end));
    
    if obj_hist(iter + 1) < obj_val
        best_controller.K = K;
        best_controller.f = f;
        
        obj_val = obj_hist(iter + 1);
    end
end
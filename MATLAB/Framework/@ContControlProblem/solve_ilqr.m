function [controller, obj_val, obj_hist] = solve_ilqr(Obj)
    n = Obj.Parameters.NStates;
    m = Obj.Parameters.NInputs;
    horizon = Obj.Parameters.Horizon;

    A = zeros(n, n, horizon);
    B = zeros(n, m, horizon);
    K = zeros(m, n, horizon);
    f = zeros(m, horizon);
    Q = zeros(n, n, horizon);
    R = zeros(m, m, horizon);
    
    P = zeros(n, n, horizon + 1);
    b = zeros(n, horizon + 1);
    
    g = Obj.Parameters.Goals;

    states = [Obj.Init.mean zeros(n, horizon)];
    nominal_states = zeros(n, horizon + 1);
    inputs = Obj.Parameters.NomInputs;

    obj_val = inf;
    obj_hist = zeros(Obj.SolverOptions.Iters, 1);
    
    for iter = 1:Obj.SolverOptions.Iters
        % Forward Equations
        for t = 1:horizon
            inputs(:, t) = inputs(:, t) + K(:, :, t) * (states(:, t) - nominal_states(:, t)) + f(:, t);
            [A(:, :, t), B(:, :, t)] = linearize(Obj, states(:, t), inputs(:, t), t);
            [Q(:, :, t), R(:, :, t)] = quadraticize_cost(Obj, states(:, t), inputs(:, t), t);
            
            states(:, t + 1) = dynamics(Obj, states(:, t), inputs(:, t), t);
            obj_hist(iter) = obj_hist(iter) + cost(Obj, states(:, t), inputs(:, t), t);
        end
        
        nominal_states = states;
        
        obj_hist(iter) = obj_hist(iter) + terminal_cost(Obj, states(:, end));
        
        fprintf('\t[%d]\tObjective: %f\n', iter, obj_hist(iter));
        
        % Backward Equations
        P(:, :, end) = quadraticize_terminal_cost(Obj, states(:, end));
        b(:, end) = P(:, :, end) * (states(:, end) - g(:, end));
        
        for t = horizon:-1:1
            K(:, :, t) = -inv(R(:, :, t) + B(:, :, t)' * P(:, :, t + 1) * B(:, :, t)) ...
                * B(:, :, t)' * P(:, :, t + 1) * A(:, :, t);
            
            f(:, t) = -(R(:, :, t) + B(:, :, t)' * P(:, :, t + 1) * B(:, :, t)) ...
                \ (B(:, :, t)' * b(:, t + 1) + R(:, :, t) * inputs(:, t));
            
            P(:, :, t) = A(:, :, t)' * P(:, :, t + 1) ...
                * (A(:, :, t) - B(:, :, t) * K(:, :, t)) + Q(:, :, t);
            
            b(:, t) = (A(:, :, t) - B(:, :, t) * K(:, :, t))' * b(:, t + 1) ...
                - K(:, :, t)' * R(:, :, t) * inputs(:, t) + Q(:, :, t) * states(:, t);
        end
        
        if obj_hist(iter) < obj_val
            controller.K = K;
            controller.f = f;
            controller.A = A;
            controller.B = B;
            controller.nominal_states = nominal_states;
            controller.nominal_inputs = inputs;
        end
    end
    
    Obj.SolverName = 'Exact';
    Obj.Controller = controller;
end


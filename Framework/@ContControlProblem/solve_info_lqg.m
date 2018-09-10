function [controller, obj_val, obj_hist, mean_traj, mean_inputs] = solve_info_lqg(Obj)
%SOLVE_INFO Summary of this function goes here
%   Detailed explanation goes here

n = Obj.Parameters.NStates;
m = Obj.Parameters.NInputs;
p = Obj.SolverOptions.NumCodewords;
horizon = Obj.Parameters.Horizon;

if isinf(Obj.Parameters.Horizon)
    [controller, obj_val, obj_hist] = solve_info_inf(Obj);
else
    if Obj.SolverOptions.FixedCodeMap
        [controller, obj_val, obj_hist] = solve_info_finite_fixed(Obj);
    else
        [controller, obj_val, obj_hist, mean_traj, mean_inputs] = solve_info_finite(Obj);
    end
end

Obj.Controller = controller;
Obj.SolverName = 'Info';

end

function [controller, obj_val, obj_hist] = solve_info_inf(Obj)
    error('Not yet implemented');
end

function [controller, obj_val, obj_hist] = solve_info_finite_fixed(Obj)
    error('Not yet implemented');
end

function [controller, obj_val, obj_hist, mean_traj, mean_inputs] = solve_info_finite(Obj)
    n = Obj.Parameters.NStates;
    m = Obj.Parameters.NInputs;
    p = Obj.SolverOptions.NumCodewords;
    horizon = Obj.Parameters.Horizon;
    tradeoff = Obj.SolverOptions.Tradeoff;

    
    A = zeros(n, n, horizon);
	B = zeros(n, m, horizon);
    Q = zeros(n, n, horizon + 1);
    R = zeros(m, m, horizon);
    f = zeros(m, horizon);       
    
    if isfield(Obj.SolverOptions, 'InitController')
        C =  Obj.SolverOptions.InitController.C;
        d =  Obj.SolverOptions.InitController.d;
        Sigma_eta =  Obj.SolverOptions.InitController.Sigma_eta;
        K =  Obj.SolverOptions.InitController.K;
    else
        C = rand(p, n, horizon);
        d = rand(p, horizon);
        K = rand(m, p, horizon);        

        for t = 1:horizon
            Sigma_eta(:, :, t) = rand(p, p);
            Sigma_eta(:, :, t) = (Sigma_eta(:, :, t) * Sigma_eta(:, :, t)');
        end
    end
    
    P = zeros(n, n, horizon + 1);
    b = zeros(n, horizon + 1);
    
    g = Obj.Parameters.Goals;
    w = Obj.Parameters.NomInputs;

    states = Obj.Init;
    inputs = rand(m, horizon);

    obj_val = inf;
    obj_hist = zeros(Obj.SolverOptions.Iters, 1);
    mi_total = 0;
    expected_cost_total = 0;
    
    mean_inputs = zeros(m, horizon);

    for iter = 1:Obj.SolverOptions.Iters
        [states, inputs, A, B, Q, R, expected_cost_total, mi_total] = forward_equations(Obj, states, C, d, Sigma_eta, K, f, horizon);
        
        obj_hist(iter) = expected_cost_total + (1 / tradeoff) * mi_total;
        
        fprintf('\t[%d]\tExpected Cost: %f\tMI: %f\tTotal: %f\n', iter, expected_cost_total, mi_total, obj_hist(iter));

        if obj_hist(iter) < obj_val
            obj_val = obj_hist(iter);
            controller.C = C;
            controller.d = d;
            controller.K = K;
            controller.Sigma_eta = Sigma_eta;
            controller.f = f;
            
            mean_traj = states;
            mean_inputs = inputs;
        end

        % Backward Equations
        P(:, :, end) = Q(:, :, end);
        b(:, end) = -Q(:, :, end) * g(:, end);
        
        for t = horizon:-1:1
            [C(:, :, t), d(:, t), Sigma_eta(:, :, t)] = solve_code_given_state(states(t), A(:, :, t), B(:, :, t), C(:, :, t), d(:, t), ...
                Sigma_eta(:, :, t), K(:, :, t), f(:, t), R(:, :, t), w(:, t), P(:, :, t + 1), b(:, t + 1), tradeoff);

            [K(:, :, t), f(:, t)] = solve_input_given_code(states(t), A(:, :, t), B(:, :, t), C(:, :, t), d(:, t), ...
                Sigma_eta(:, :, t), K(:, :, t), f(:, t), R(:, :, t), w(:, t), P(:, :, t + 1), b(:, t + 1), Obj.SolverOptions.Tol);
            
            [P(:, :, t), b(:, t)] = solve_value_function(states(t), A(:, :, t), B(:, :, t), C(:, :, t), d(:, t), Sigma_eta(:, :, t), ...
                K(:, :, t), f(:, t), Q(:, :, t), R(:, :, t), g(:, t), w(:, t), P(:, :, t + 1), b(:, t + 1), tradeoff);
        end
    end
    
    [~, ~, ~, ~, ~, ~, expected_cost_total, mi_total] = forward_equations(Obj, states, C, d, Sigma_eta, K, f, horizon);
        
    obj_hist(iter) = expected_cost_total + (1 / tradeoff) * mi_total;

    fprintf('\t[%d]\tExpected Cost: %f\tMI: %f\tTotal: %f\n', iter, expected_cost_total, mi_total, obj_hist(iter));

    if obj_hist(iter) < obj_val
        obj_val = obj_hist(iter);
        controller.C = C;
        controller.d = d;
        controller.K = K;
        controller.Sigma_eta = Sigma_eta;
        controller.f = f;
        
        mean_traj = states;
        mean_inputs = inputs;
    end
end


function [states, inputs, A, B, Q, R, expected_cost_total, mi_total] = forward_equations(Obj, states, C, d, Sigma_eta, K, f, horizon)
    mi_total = 0;
    expected_cost_total = 0;

    % Forward equations
    for t = 1:horizon
        inputs(:, t) = K(:, :, t) * (C(:, :, t) * states(t).mean + d(:, t)) + f(:, t);
        input_cov = K(:, :, t) * (C(:, :, t) * states(t).cov * C(:, :, t)' + Sigma_eta(:, :, t)) * K(:, :, t)';

        [A(:, :, t), B(:, :, t)] = linearize(Obj, states(t).mean, inputs(:, t));
        [Q(:, :, t), R(:, :, t)] =  quadraticize_cost(Obj, states(t).mean, inputs(:, t));

        states(t + 1).mean = dynamics(Obj, states(t).mean, inputs(:, t));
        states(t + 1).cov = A(:, :, t) * states(t).cov * A(:, :, t)' ...
        + B(:, :, t) * input_cov * B(:, :, t)'...
        + Obj.Parameters.ProcCov;

        expected_cost_total = expected_cost_total ....
            + cost(Obj, states(t).mean, inputs(:, t), t) + trace(Q(:, :, t) * states(t).cov) ...
            + trace(R(:, :, t) * input_cov);

        mi_total = mi_total + mutual_info(states(t).cov, C(:, :, t), Sigma_eta(:, :, t));
    end

    Q(:, :, end) = quadraticize_terminal_cost(Obj, states(end).mean);

    expected_cost_total = expected_cost_total + terminal_cost(Obj, states(end).mean) + trace(Q(:, :, end) * states(end).cov);
end




function [C, d, Sigma_eta] = solve_code_given_state(state, A, B, C, d, Sigma_eta, K, f, R, w, P, b, tradeoff)
    Sigma_eta = inv(tradeoff * K' * (B' * P * B + R) * K ... 
       + inv(C * state.cov * C' + Sigma_eta));

    F = inv(C * state.cov * C' + Sigma_eta);

    C = -tradeoff * Sigma_eta * K' * B' * P * A;

    d = -Sigma_eta * (tradeoff * K' * B' * (b + P * B * f) + tradeoff * K' * R * (f - w) - F * (C * state.mean + d));
end

function [K, f] = solve_input_given_code(state, A, B, C, d, Sigma_eta, K, f, R, w, P, b, tol)
    n = size(A, 1);
    m = size(B, 2);
    p = size(C, 1);
    
    x_bar = state.mean;
    Sigma_x = state.cov;
    
    x_tilde_bar = C * x_bar + d;
    Sigma_x_tilde = C * Sigma_x * C' + Sigma_eta;
    
    K = sdpvar(m, p, 'full');
    
    constraint = [R * K * x_tilde_bar * x_tilde_bar' + R * K * Sigma_x_tilde ...
        + B' * P * A * x_bar * x_tilde_bar' + B' * P * B * K * x_tilde_bar * x_tilde_bar' ...
        + B' * P * B * K * Sigma_x_tilde + B' * b * x_tilde_bar' ...
        + R * f * x_tilde_bar' - R * w * x_tilde_bar' + B' * P * B * f * x_tilde_bar' <= tol;
        
        R * K * x_tilde_bar * x_tilde_bar' + R * K * Sigma_x_tilde ...
        + B' * P * A * x_bar * x_tilde_bar' + B' * P * B * K * x_tilde_bar * x_tilde_bar' ...
        + B' * P * B * K * Sigma_x_tilde + B' * b * x_tilde_bar' ...
        + R * f * x_tilde_bar' - R * w * x_tilde_bar' + B' * P * B * f * x_tilde_bar'  >= -tol];
    
    options = sdpsettings('verbose', false, 'debug', true);
    
    sol = optimize(constraint, 0, options);
    
    if sol.problem == 0
        % Extract and display value
        K = value(K);
    else
        sol.info
        yalmiperror(sol.problem)
        error('Error satisfying linear controller constraint!!!');
    end
    
    f = sdpvar(m, 1, 'full');
    
    objective = (K * x_tilde_bar + f)' * R * (K * x_tilde_bar + f) ...
        + (A * state.mean + B * K * x_tilde_bar + B * f)' * P * (A * state.mean + B * K * x_tilde_bar + B * f) ...
        + b' * (A * state.mean + B * K * x_tilde_bar + B * f);
    
    sol = optimize([], objective, options);
    
%     constraint = [R * K * x_tilde_bar + R * f + B' * P * (A * x_bar + B * K * x_tilde_bar + B * f) + B' * b - R * w <= tol;
%         
%                   R * K * x_tilde_bar + R * f + B' * P * (A * x_bar + B * K * x_tilde_bar + B * f) + B' * b - R * w >= -tol];
%     
%     sol = optimize(constraint, 0, options);
    
    if sol.problem == 0
        % Extract and display value
        f = value(f);
    else
        sol.info
        yalmiperror(sol.problem)
        error('Error satisfying linear controller constraint!!!');
    end
end

function [P, b] = solve_value_function(state, A, B, C, d, Sigma_eta, K, f, Q, R, g, w, P, b, tradeoff)
    F = inv(C * state.cov * C' + Sigma_eta);
    G = C' * F * C;

    P = Q + (1 / tradeoff) * G + C' * K' * R * K * C + (A + B * K * C)' * P * (A + B * K * C);
            
    b = (A + B * K * C)' * P * B * K * d - Q * g - (1 / tradeoff) * G * state.mean + C' * K' * R * K * d ...
        + (A + B * K * C)' * b + C' * K' * R * f - C' * K' * R * w + A' * P * B * f + C' * K' * B' * P * B * f;
end

function mi = mutual_info(state_cov, C, Sigma_eta)
    n = size(C, 2);
    p = size(C, 1);
    
    Sigma_joint = [state_cov, state_cov * C';
                   C * state_cov, C * state_cov * C' + Sigma_eta];
    
    Hx = 0.5 * log(det(state_cov) * (2 * pi * exp(1)) ^ n);
    Hx_tilde = 0.5 * log(det(C * state_cov * C' + Sigma_eta) * (2 * pi * exp(1)) ^ p);
    Hjoint = 0.5 * log(det(Sigma_joint) * (2 * pi * exp(1)) ^ (n + p));
    
    mi = Hx - Hjoint + Hx_tilde;
end
function [controller, obj_val, obj_hist] = solve_info_ilqr(Obj)
%SOLVE_INFO_ILQR Summary of this function goes here
%   Detailed explanation goes here
    n = Obj.Parameters.NStates;
    m = Obj.Parameters.NInputs;
    p = Obj.Parameters.NumCodewords;
    horizon = Obj.Parameters.Horizon;

    A = zeros(n, n, horizon);
    B = zeros(n, m, horizon);    
    
    C = repmat(eye(p, n), 1, 1, horizon);
    d = zeros(p, horizon);
    Sigma_eta = 0.01 * repmat(eye(p, n), 1, 1, horizon);
    
    K = 0.1 * rand(m, p, horizon);
    f = zeros(m, horizon);
    
    Q = zeros(n, n, horizon);
    R = zeros(m, m, horizon);
    
    P = zeros(n, n, horizon + 1);
    b = zeros(n, horizon + 1);
    
    g = Obj.Parameters.Goals;

    states = Obj.Init;
    nominal_states = [Obj.Init.mean zeros(n, horizon)];
    mean_inputs = Obj.Parameters.NomInputs;

    obj_val = inf;
    obj_hist = zeros(Obj.SolverOptions.Iters, 1);
    
    tradeoff = Obj.Parameters.Tradeoff;
    
    for t = 1:horizon
        nominal_states(:, t + 1) = dynamics(Obj, nominal_states(:, t), mean_inputs(:, t), t);
    end
    
    for iter = 1:Obj.SolverOptions.Iters
        mi = 0;
        
        % Forward Equations
        for t = 1:horizon
            mean_inputs(:, t) = mean_inputs(:, t) + K(:, :, t) * (C(:, :, t) * (states(t).mean - nominal_states(:, t)) + d(:, t)) + f(:, t);
            cov_inputs = K(:, :, t) * (C(:, :, t) * states(t).cov * C(:, :, t)' + Sigma_eta(:, :, t)) * K(:, :, t)';
            
            [A(:, :, t), B(:, :, t)] = linearize(Obj, states(t).mean, mean_inputs(:, t), t);
            [Q(:, :, t), R(:, :, t)] = quadraticize_cost(Obj, states(t).mean, mean_inputs(:, t), t);
            
            states(t + 1).mean = dynamics(Obj, states(t).mean, mean_inputs(:, t), t);
            states(t + 1).cov = A(:, :, t) * states(t).cov * A(:, :, t)' + Obj.Parameters.ProcCov;
            
            obj_hist(iter) = obj_hist(iter) + cost(Obj, states(t).mean, mean_inputs(:, t), t) ...
                + trace(Q(:, :, t) * states(t).cov) + trace(R(:, :, t) * cov_inputs);
            
            mi = mutual_info(states(t).cov, C(:, :, t), Sigma_eta(:, :, t));
        end
        
        nominal_states = [states.mean];
        
        P(:, :, end) = quadraticize_terminal_cost(Obj, states(end).mean);
        b(:, end) = P(:, :, end) * (states(end).mean - g(:, end));
        
        obj_hist(iter) = obj_hist(iter) + terminal_cost(Obj, states(end).mean) + trace(P(:, :, end) * states(end).cov);
        
        fprintf('\t[%d]\tExpected Objective: %f\tMI: %f\tTotal: %f\n', iter, obj_hist(iter), mi, obj_hist(iter) + mi);
        
        obj_hist(iter) = obj_hist(iter) + mi;
        
        delta_g = g - nominal_states;
        delta_w = -mean_inputs;
        
        % Backward Equations
        for t = horizon:-1:1
            [C(:, :, t), d(:, t), Sigma_eta(:, :, t)] = solve_code_given_state(states(t), A(:, :, t), B(:, :, t), ...
                C(:, :, t), d(:, t), Sigma_eta(:, :, t), K(:, :, t), f(:, t), R(:, :, t), delta_w(:, t), P(:, :, t + 1), b(:, t + 1), tradeoff);
            
            [K(:, :, t), f(:, t)] = solve_input_given_code(states(t), A(:, :, t), B(:, :, t), ...
                C(:, :, t), d(:, t), Sigma_eta(:, :, t), f(:, t), R(:, :, t), delta_w(:, t), P(:, :, t + 1), b(:, t + 1));
            
            [P(:, :, t), b(:, t)] = solve_value_function(states(t), A(:, :, t), B(:, :, t), C(:, :, t), d(:, t), ...
                Sigma_eta(:, :, t), K(:, :, t), f(:, t), Q(:, :, t), delta_g(:, t), R(:, :, t), delta_w(:, t), P(:, :, t + 1), b(:, t + 1), tradeoff);
        end
        
        if obj_hist(iter) < obj_val
            controller.K = K;
            controller.f = f;
            controller.nominal_states = nominal_states;
        end
    end
end

function [C, d, Sigma_eta] = solve_code_given_state(state, A, B, C, d, Sigma_eta, K, f, R, w, P, b, tradeoff)
    Sigma_eta = inv(tradeoff * K' * (B' * P * B + R) * K ... 
       + inv(C * state.cov * C' + Sigma_eta));

    F = inv(C * state.cov * C' + Sigma_eta);

    C = tradeoff * Sigma_eta * K' * B' * P * A;

    d = Sigma_eta * (tradeoff * K' * B' * (b + P * B * f) + tradeoff * K' * R' * (f - w)- F * (C * state.mean + d));
end

function [K, f] = solve_input_given_code(state, A, B, C, d, Sigma_eta, f, R, w, P, b)
    n = size(A, 1);
    m = size(B, 2);
    p = size(C, 2);
    
    x_bar = state.mean;
    Sigma_x = state.cov;
    
    x_tilde_bar = C * x_bar + d;
    Sigma_x_tilde = C * Sigma_x * C' + Sigma_eta;
    
    K = sdpvar(m, p, 'full');
    
    constraint = [R * K * x_tilde_bar * x_tilde_bar' + R * f * x_tilde_bar' + R * K * Sigma_x_tilde ...
        + B' * P * A * x_bar * x_tilde_bar' + B' * P * B * K * x_tilde_bar * x_tilde_bar' ...
        + B' * P * B * f * x_tilde_bar' + B' * P * B * K * Sigma_x_tilde + B' * b * x_tilde_bar' - R * w * x_tilde_bar' == 0];
    
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
    
    constraint = [R * K * x_tilde_bar + R * f + B' * P * A * x_bar ...
        + B' * P * B * K * x_tilde_bar + B' * P * B * f + b' * B - R * w == 0];
    
    options = sdpsettings('verbose', false, 'debug', true);
    
    sol = optimize(constraint, 0, options);
    
    if sol.problem == 0
        % Extract and display value
        f = value(f);
    else
        sol.info
        yalmiperror(sol.problem)
        error('Error satisfying linear controller constraint!!!');
    end
    
end

function [P, b] = solve_value_function(state, A, B, C, d, Sigma_eta, K, f, Q, g, R, w, P, b, tradeoff)
    F = inv(C * state.cov * C' + Sigma_eta);
    G = C' * F * C;

    P = Q + (1 / tradeoff) * G + C' * K' * R * K * C + (A + B * K * C)' * P * (A + B * K * C);
            
    b = (A + B * K * C)' * P * B * K * d - Q * g - (1 / tradeoff) * G * state.mean + C' * K' * R * K * d + b ...
        + C' * K' * R * f - C' * K' * R * w + A' * P * B * f + C' * K' * B' * P * B * f;
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
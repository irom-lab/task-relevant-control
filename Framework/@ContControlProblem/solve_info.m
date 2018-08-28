function [controller, obj_val, obj_hist] = solve_info(Obj)
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
        [controller, obj_val, obj_hist] = solve_info_finite(Obj);
    end
end

Obj.Controller = controller;

end

function [controller, obj_val, obj_hist] = solve_info_inf(Obj)
    error('Not yet implemented');
end

function [controller, obj_val, obj_hist] = solve_info_finite_fixed(Obj)
    error('Not yet implemented');
end

function [controller, obj_val, obj_hist] = solve_info_finite(Obj)
    n = Obj.Parameters.NStates;
    m = Obj.Parameters.NInputs;
    p = Obj.SolverOptions.NumCodewords;
    horizon = Obj.Parameters.Horizon;
    tradeoff = Obj.SolverOptions.Tradeoff;

    A = zeros(n, n, horizon);
    B = zeros(n, m, horizon);
    C = zeros(p, n, horizon);
    d = rand(p, horizon);
    K = rand(m, p, horizon);
    
    for t = 1:horizon
        Sigma_eta(:, :, t) = rand(p, p);
        Sigma_eta(:, :, t) = (Sigma_eta(:, :, t) * Sigma_eta(:, :, t)');
    end
    
    P = zeros(n, n, horizon + 1);
    b = zeros(n, horizon + 1);
    
    g = Obj.Parameters.Goals;

    states = Obj.Init;
    inputs = rand(m, horizon);

    obj_val = inf;
    obj_hist = zeros(Obj.SolverOptions.Iters, 1);

    for iter = 1:Obj.SolverOptions.Iters
        % Forward equations
        for t = 1:horizon
            inputs(:, t) = K(:, :, t) * C(:, :, t) * states(t).mean;
            [A(:, :, t), B(:, :, t)] = linearize(Obj, states(t).mean, inputs(:, t));

            states(t + 1).mean = dynamics(Obj, states(t).mean, inputs(:, t));
            states(t + 1).cov = A(:, :, t) * states(t).cov * A(:, :, t)' + Obj.Parameters.ProcCov;

            obj_hist(iter) = obj_hist(iter) + cost(Obj, states(t).mean, inputs(:, t)) + ...
                (1 / tradeoff) * mutual_info(states(t).cov, C(:, :, t), Sigma_eta(:, :, t));
        end

        obj_hist(iter) = obj_hist(iter) + terminal_cost(Obj, states(end).mean);

        if obj_hist(iter) < obj_val
            obj_val = obj_hist(iter);
            controller.C = C;
            controller.d = d;
            controller.K = K;
            controller.Sigma_eta = Sigma_eta;
        end

        % Backward Equations
        P(:, :, end) = quadraticize_terminal_cost(Obj, states(end).mean);
        b(:, end) = -P(:, :, end) * Obj.Parameters.Goals(:, end);
        
        for t = horizon:-1:1
            [Q, R] = quadraticize_cost(Obj, states(t).mean, inputs(:, t));
            
            [C(:, :, t), d(:, t), Sigma_eta(:, :, t)] = solve_code_given_state(states(t), A(:, :, t), B(:, :, t), C(:, :, t), d(:, t), ...
                Sigma_eta(:, :, t), K(:, :, t), R, P(:, :, t + 1), b(:, t + 1), tradeoff);

            K(:, :, t) = solve_input_given_code(states(t), A(:, :, t), B(:, :, t), C(:, :, t), d(:, t), ...
                Sigma_eta(:, :, t), R, P(:, :, t + 1), b(:, t + 1));
            
            [P(:, :, t), b(:, t)] = solve_value_function(states(t), A(:, :, t), B(:, :, t), C(:, :, t), d(:, t), Sigma_eta(:, :, t), ...
                K(:, :, t), Q, R, g(:, t), P(:, :, t + 1), b(:, t + 1), tradeoff);
        end
    end
    
    obj_hist(end + 1) = 0;

    % Forward equations
    for t = 1:horizon
        [A(:, :, t), B(:, :, t)] = linearize(Obj, states(t).mean, inputs(:, t));

        states(t + 1).mean = dynamics(Obj, states(t).mean, inputs(:, t));
        states(t + 1).cov = A(:, :, t) * states(t).cov * A(:, :, t)' + Obj.Parameters.ProcCov;

        obj_hist(end) = obj_hist(end) + cost(Obj, states(t).mean, inputs(:, t)) + ...
            (1 / tradeoff) * mutual_info(states(t).cov, C(:, :, t), Sigma_eta(:, :, t));
    end

    obj_hist(end) = obj_hist(end) + terminal_cost(Obj, states(end).mean);

    if obj_hist(end) < obj_val
        obj_val = obj_hist(end + 1);
        controller.C = C;
        controller.d = d;
        controller.K = K;
        controller.P = P;
    end
end

function [C, d, Sigma_eta] = solve_code_given_state(state, A, B, C, d, Sigma_eta, K, R, P, b, tradeoff)
    Sigma_eta = inv(tradeoff * K' * (B' * P * B + R) * K ... 
       + inv(C * state.cov * C' + Sigma_eta));

    F = inv(C * state.cov * C' + Sigma_eta);

    C = tradeoff * Sigma_eta * K' * B' * P * A;

    d = Sigma_eta * (tradeoff * K' * B' * b + F * (C * state.mean + d));
end

function K_val = solve_input_given_code(state, A, B, C, d, Sigma_eta, R, P, b)
    n = size(A, 1);
    m = size(B, 2);
    p = size(C, 2);
    
    x_bar = state.mean;
    Sigma_x = state.cov;
    
    x_tilde_bar = C * x_bar + d;
    Sigma_x_tilde = C * Sigma_x * C' + Sigma_eta;
    
    K = sdpvar(m, p, 'full');
    
    constraint = [R * K * x_tilde_bar * x_tilde_bar' + R * K * Sigma_x_tilde ...
        + B' * P * A * x_bar * x_tilde_bar' + B' * P * B * K * x_tilde_bar * x_tilde_bar' ...
        + B' * P * B * K * Sigma_x_tilde + B' * b * x_tilde_bar' == 0];
    
    options = sdpsettings('verbose', false, 'debug', true);
    
    sol = optimize(constraint, 0, options);
    
    if sol.problem == 0
        % Extract and display value
        K_val = value(K);
    else
        sol.info
        yalmiperror(sol.problem)
        error('Error satisfying linear controller constraint!!!');
    end
    
end

function [P, b] = solve_value_function(state, A, B, C, d, Sigma_eta, K, Q, R, g, P, b, tradeoff)
    F = inv(C * state.cov * C' + Sigma_eta);
    G = C' * F * C;

    P = Q + (1 / tradeoff) * G + C' * K' * R * K * C + (A + B * K * C)' * P * (A + B * K * C);
            
    b = A' * P * B * K * d - Q * g - (1 / tradeoff) * G * state.mean - C' * K' * R * K * d + b;
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
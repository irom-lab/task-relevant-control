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
    tradeoff = Obj.SolverOptions.Tradeoff

    A = zeros(n, n, horizon);
    B = zeros(n, m, horizon);
    C = zeros(p, n, horizon);
    d = rand(p, horizon);
    K = rand(m, p, horizon);
    Sigma_eta = rand(p, p, horizon);
    P = zeros(n, n, horizon + 1);
    b = zeros(n, horizon + 1);

    states = Obj.Init;
    inputs = rand(m, horizon);

    obj_val = inf;
    obj_hist = zeros(Obj.SolverOptions.Iters, 1);

    for iter = 1:Obj.SolverOptions.Iters
        % Forward equations
        for t = 1:horizon
            [A(:, :, t), B(:, :, t)] = linearize(Obj, states(t).mean, inputs(:, t));
            inputs(:, t) = K(:, :, t) * C(:, :, t) * states(t).mean;

            states(t + 1).mean = dynamics(Obj, states(t).mean, inputs(:, t));
            states(t + 1).cov = A(:, :, t) * states(t).cov * A(:, :, t)' + Obj.Parameters.ProcNoise;

            obj_hist(iter) = obj_hist(iter) + cost(Obj, states(t).mean, inputs(:, t)) + ...
                (1 / tradeoff) * mutual_info(states(t).mean, states(t).cov, C(:, :, t) * states(t).mean, C(:, :, t) * states(t).cov * C(:, :, t)');
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
        b(:, :, end) = -P(:, :, end) * Obj.Parameters.Goals(:, end);
        
        for t = horizon:-1:1
            [Q, R] = quadraticize_cost(Obj, states(t).mean, inputs(:, t));
            
            Sigma_eta(:, :, t) = inv(inv(C(:, :, t) * state(t).cov * C(:, :, t)' + Sigma_eta(:, :, t)) ...
                - tradeoff * K(:, :, t)' * (B(:, :, t)' * P(:, :, t) * B(:, :, t) + R) * K(:, :, t));
            
            C(:, :, t) = (1/2) * inv(Sigma_eta(:, :, t)) * (tradeoff * K(:, :, t)'* B(:, :, t) * P(:, :, t + 1) * A(:, :, t) ...
                - inv(C(:, :, t) * state(t).cov * C(:, :, t)' + Sigma_eta(:, :, t)) * d(:, t));
            
            d(:, :, t) = (1/2) * inv(Sigma_eta(:, :, t)) * (tradeoff * K(:, :, t)' * B(:, :, t)' * b(:, t + 1) ...
                + inv(C(:, :, t) * state(t).cov * C(:, :, t)' + Sigma_eta(:, :, t)) * d(:, t));

            K(:, :, t) = solve_csp();
            
            
            G = C(:, :, t)' * inv(C(:, :, t) * state(t).cov * C(:, :, t)' + Sigma_eta(:, :, t)) * C(:, :, t);
            
            P(:, :, t) = Q + (1 / tradeoff) * G + C(:, :, t)' * K(:, :, t)' * R * K(:, :, t) * C(:, :, t) ...
                + (A(:, :, t) + B(:, :, t) * C(:, :, t))' * P(:, :, t + 1) * (A(:, :, t) + B(:, :, t) * C(:, :, t));
            
            b(:, t) = A(:, :, t)' * P(:, :, t) * B(:, :, t) * K(:, :, t) * d(:, :, t) - Q * Obj.Parameters.Goals(:, t) ...
                - G * state(t).mean - C(:, :, t)' * K(:, :, t)' * R * K(:, :, t) * d(:, t) + b(:, t + 1);
        end
    end

    % Forward equations
    for t = 1:horizon
        [A(:, :, t), B(:, :, t)] = linearize(Obj, states(t).mean, inputs(:, t));

        states(t + 1).mean = dynamics(Obj, states(t).mean, inputs(:, t));
        states(t + 1).cov = A(:, :, t) * states(t).cov * A(:, :, t)' + Obj.Parameters.ProcNoise;



        obj_hist(iter) = obj_hist(iter) + cost(Obj, states(t).mean, inputs(:, t)) + ...
            beta * mutual_info(states(t).mean, states(t).cov, C(:, :, t) * states(t).mean, C(:, :, t) * states(t).cov * C(:, :, t));
    end

    obj_hist(iter) = obj_hist(iter) + terminal_cost(Obj, states(end).mean);

    if obj_hist(iter) < obj_val
        obj_val = obj_hist(iter);
        controller.C = C;
        controller.d = d;
        controller.K = K;
        controller.P = P;
    end
end

function K = solve_csp()
end

function mi = mutual_info(mean1, mean2, full_cov)
    n = length(mean1);
    
    cov1 = full_cov(1:n, 1:n);
    h1 = (1 / 2) * log(((2 * pi * exp(1))^n) * det(cov1));
    
    mi = (1 / 2) * log(det(full_cov(1:n, 1:n)) * det(full_cov((n + 1):end, (n + 1):end)) / det(full_cov));
end
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

    A = zeros(n, n, horizon);
    B = zeros(n, m, horizon);
    C = zeros(p, n, horizon);
    d = rand(p, horizon);
    K = rand(m, p, horizon);
    Sigma_eta = rand(p, p, horizon);
    P = zeros(n, n, horizon + 1);

    states = Obj.Init;
    inputs = rand(m, horizon);

    obj_val = inf;
    obj_hist = zeros(Obj.SolverOptions.Iters, 1);

    for iter = 1:Obj.SolverOptions.Iters
        % Forward equations
        for t = 1:horizon
            [A(:, :, t), B(:, :, t)] = linearize(Obj, states(t).mean, inputs(:, t));

            states(t + 1).mean = dynamics(Obj, states(t).mean, inputs(:, t));
            states(t + 1).cov = A(:, :, t) * states(t).cov * A(:, :, t)' + Obj.Parameters.ProcNoise;

            obj_hist(iter) = obj_hist(iter) + cost(Obj, states(t).mean, inputs(:, t)) + ...
                mutual_info(states(t).mean, states(t).cov, C(:, :, t) * states(t).mean, C(:, :, t) * states(t).cov * C(:, :, t));
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

        for t = horizon:-1:1
            C(:, :, t) = ;
            d(:, :, t) = ;
            Sigma_eta(:, :, t) = ;

            K = solve_csp();
            P = ;
        end
    end

    % Forward equations
    for t = 1:horizon
        [A(:, :, t), B(:, :, t)] = linearize(Obj, states(t).mean, inputs(:, t));

        states(t + 1).mean = dynamics(Obj, states(t).mean, inputs(:, t));
        states(t + 1).cov = A(:, :, t) * states(t).cov * A(:, :, t)' + Obj.Parameters.ProcNoise;



        obj_hist(iter) = obj_hist(iter) + cost(Obj, states(t).mean, inputs(:, t)) + ...
            mutual_info(states(t).mean, states(t).cov, C(:, :, t) * states(t).mean, C(:, :, t) * states(t).cov * C(:, :, t));
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
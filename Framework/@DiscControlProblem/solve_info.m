function [ controller, obj_val, obj_hist ] = solve_info(Obj)
%SOLVE_INFO Summary of this function goes here
%   Detailed explanation goes here

horizon = Obj.Parameters.Horizon;
iters = Obj.SolverOptions.Iters;

[n, ~, m] = size(Obj.Transitions);
p = Obj.SolverOptions.NumCodewords;

tradeoff = Obj.SolverOptions.Tradeoff;

if isinf(horizon)
    error('Not Implemented');
    
    for iter = 1:iters
        
    end
else
    values = [zeros(n, horizon) Obj.TerminalCosts];
    
    if isfield(Obj.SolverOptions, 'InitPolicy')
        code_given_state = Obj.SolverOptions.InitPolicy.CodeGivenState;
        input_given_code = Obj.SolverOptions.InitPolicy.InputGivenCode;
    else
        code_given_state = rand(p, n, horizon);
        code_given_state = code_given_state ./ sum(code_given_state);
    
        input_given_code = rand(m, p, horizon);
        input_given_code = input_given_code ./ sum(input_given_code);
    end
    
    if isfield(Obj.SolverOptions, 'FixedCodeMap') && Obj.SolverOptions.FixedCodeMap
        code_given_state = code_given_state(:, :, 1);
        fixed_code_map = true;
    else
        fixed_code_map = false;
    end
    
    state_dist = [Obj.Init zeros(n, horizon - 1)];
    code_dist = zeros(p, horizon - 1);
    
    obj = objective(Obj, code_given_state, input_given_code, Obj.Init);
    options = optimoptions(@linprog, 'Algorithm', 'dual-simplex', 'Display', 'off');
    
    best_code_given_state = zeros(p, n);
    best_input_given_code = zeros(m, p);
    obj_val = Inf;
    
    for iter = 1:iters
        % Forward Equations
        for t = 1:horizon
            if fixed_code_map
                input_given_state = input_given_code(:, :, t) * code_given_state;
                T = forward_eqn(Obj.Transitions, input_given_state);
                state_dist(:, t + 1) = T * state_dist(:, t);
                code_dist(:, t) = code_given_state * state_dist(:, t);
            else
                input_given_state = input_given_code(:, :, t) * code_given_state(:, :, t);
                T = forward_eqn(Obj.Transitions, input_given_state);
                state_dist(:, t + 1) = T * state_dist(:, t);
                code_dist(:, t) = code_given_state(:, :, t) * state_dist(:, t);
            end
        end
        
        % Backward Equations
        for t = horizon:-1:1
            % Code given state
            for i = 1:n
                for j = 1:p
                    exponent = tradeoff * (values(:, t + 1)' * (squeeze(Obj.Transitions(:, i, :)) * input_given_code(:, j, t)) ...
                        - Obj.Costs(i, :) * input_given_code(:, j, t));
                    
                    if fixed_code_map
                        code_given_state(j, i) = code_dist(j) * exp(exponent);
                    else
                        code_given_state(j, i, t) = code_dist(j) * exp(exponent);
                    end
                end
            end
            
            if fixed_code_map
                code_given_state(:, :) = code_given_state(:, :) ./ sum(code_given_state(:, :));
            else
                code_given_state(:, :, t) = code_given_state(:, :, t) ./ sum(code_given_state(:, :, t));
            end
            
            % Input Given Code
            for i = 1:p
                c = zeros(1, m);
                Aeq = ones(1, m);
                beq = 1;
                
                for j = 1:m
                    if fixed_code_map
                        c(j) = code_given_state(i, :) * (Obj.Costs(:, j) .* state_dist(:, t));
                        c(j) = c(j) + values(:, t + 1)' * Obj.Transitions(:, :, j) * (code_given_state(i, :)' .* state_dist(:, t));
                    else
                        c(j) = code_given_state(i, :, t) * (Obj.Costs(:, j) .* state_dist(:, t));
                        c(j) = c(j) + values(:, t + 1)' * Obj.Transitions(:, :, j) * (code_given_state(i, :, t)' .* state_dist(:, t));
                    end
                end
                
                input_given_code(:, i, t) = linprog(c, [], [], Aeq, beq, zeros(m, 1), ones(m, 1), options);
            end
            
            % Value function
            for i = 1:n
                if fixed_code_map
                    input_given_state = input_given_code(:, :, t) * code_given_state(:, i);
                    code_dist(:, t) = code_given_state * state_dist(:, t);

                    values(i, t) = Obj.Costs(i, :) * input_given_state ...
                        + values(:, t + 1)' * (squeeze(Obj.Transitions(:, i, :)) * input_given_state) ...
                        + (1 / tradeoff) * kl_div(code_given_state(:, i), code_dist(:, t));
                else
                    input_given_state = input_given_code(:, :, t) * code_given_state(:, i, t);
                    code_dist(:, t) = code_given_state(:, :, t) * state_dist(:, t);
                    
                     values(i, t) = Obj.Costs(i, :) * input_given_state ...
                         + values(:, t + 1)' * (squeeze(Obj.Transitions(:, i, :)) * input_given_state) ...
                         + (1 / tradeoff) * kl_div(code_given_state(:, i, t), code_dist(:, t));

                    
                end
            end
        end
        
        obj = [obj; objective(Obj, code_given_state, input_given_code, Obj.Init)];
        fprintf('\t[%d]\tObjective: %f\n', iter, obj(end));
            
        if obj(end) < obj_val
            obj_val = obj(end);
            Obj.Controller.CodeGivenState = code_given_state;
            Obj.Controller.InputGivenCode = input_given_code;
            

            for t = 1:horizon
                if fixed_code_map
                    Obj.Controller.Policy(:, :, t) = input_given_code(:, :, t) * code_given_state;
                else
                    Obj.Controller.Policy(:, :, t) = input_given_code(:, :, t) * code_given_state(:, :, t);
                end
            end
        end
    end
end

Obj.SolverName = 'Info';
Obj.Controller.Estimator = 'State';
controller = Obj.Controller.Policy;
obj_hist = obj;

end

function kl = kl_div(a, b)
    pointwise = a .* log(a ./ b);
    pointwise(isnan(pointwise)) = 0;
    kl = sum(pointwise);
end

function info = mutual_info(a, b, joint)
    pointwise_info = joint .* log(joint ./ (a .* b'));
    pointwise_info(~isfinite(pointwise_info)) = 0;
    
    info = sum(pointwise_info(:));
end

function obj = objective(Obj, code_given_state, input_given_code, init)
    obj = 0;
    mi = 0;
    state_dist = init;
    
    for t = 1:Obj.Parameters.Horizon
        if size(code_given_state, 3) == 1
            input_dist = input_given_code(:, :, t) * code_given_state * state_dist;
            state_input = ((input_given_code(:, :, t) * code_given_state) .* state_dist')';
            
            code_dist = code_given_state * state_dist;
            code_state = code_given_state .* state_dist';            
            
            T = forward_eqn(Obj.Transitions, input_given_code(:, :, t) * code_given_state);
        else
            input_dist = input_given_code(:, :, t) * code_given_state(:, :, t) * state_dist;
            state_input = ((input_given_code(:, :, t) * code_given_state(:, :, t)) .* state_dist')';
            
            code_dist = code_given_state(:, :, t) * state_dist;
            code_state = code_given_state(:, :, t) .* state_dist';            
            
            T = forward_eqn(Obj.Transitions, input_given_code(:, :, t) * code_given_state(:, :, t));
        end        
        
        obj = obj + sum(Obj.Costs(:) .* state_input(:));
        
        mi_pointwise = code_state .* log(code_state ./ (code_dist' .* state_dist)');
        mi_pointwise(isnan(mi_pointwise) | isinf(mi_pointwise)) = 0;
        mi = mi + sum(mi_pointwise(:));
        
        state_dist = T * state_dist;
    end
    
    obj = obj + Obj.TerminalCosts' * state_dist;
    
    obj = obj + (1 / Obj.SolverOptions.Tradeoff) * mi;
end

function T = forward_eqn(transitions, policy)
    T = zeros(size(policy, 2));
    
    for i = 1:size(policy, 2)
        T(:, i) = squeeze(transitions(:, i, :)) * policy(:, i);
    end
end
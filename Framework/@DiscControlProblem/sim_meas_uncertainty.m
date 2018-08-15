function [ traj, cum_cost ] = sim_meas_uncertainty(Obj, init_dist, horizon)
%SIMULATE_INIT_UNCERTAINTY Summary of this function goes here
%   Detailed explanation goes here

if isinf(Obj.Parameters.Horizon) && nargin < 3
    error('Control problem is infinite horizon. Specify horizon for sim.');
elseif isfinite(Obj.Parameters.Horizon)
    horizon = Obj.Parameters.Horizon;
end

n = size(Obj.Transitions, 1);
m = size(Obj.Transitions, 3);

if isequal(Obj.SolverName, 'Info')
    p = Obj.SolverOptions.NumCodewords;
end

traj = [discretesample(init_dist, 1); zeros(horizon, 1)];
costs = zeros(horizon + 1, 1);

state_dist = init_dist;
output_given_state = Obj.Sensor;

if isequal(Obj.SolverName, 'Info') && isequal(Obj.Controller.Estimator, 'Code')
    state_dist = [init_dist zeros(n, horizon)];
    code_dist = [Obj.Controller.CodeGivenState(:, :, 1) * init_dist zeros(p, horizon)];
    bayes_transitions = zeros(p, p, m, horizon);
    
    state_given_code = zeros(n, p, horizon);
    state_given_code(:, :, 1) = ((Obj.Controller.CodeGivenState(:, :, 1) .* state_dist(:, 1)') ./ code_dist(:, 1))';
    
    if isfinite(horizon)
        for t = 1:(horizon - 1)
            T = forward_eqn(Obj.Transitions, Obj.Controller.Policy);
            state_dist(:, t + 1) = T * state_dist(:, t);
            
            if size(Obj.Controller.CodeGivenState, 3) == 1
                code_dist(:, t + 1) = Obj.Controller.CodeGivenState * state_dist(:, t + 1);
                state_given_code(:, :, t + 1) = ((Obj.Controller.CodeGivenState .* state_dist(:, t + 1)') ./ code_dist(:, t + 1))';
            else
                code_dist(:, t + 1) = Obj.Controller.CodeGivenState(:, :, t + 1) * state_dist(:, t + 1);
                state_given_code(:, :, t + 1) = ((Obj.Controller.CodeGivenState(:, :, t + 1) .* state_dist(:, t + 1)') ./ code_dist(:, t + 1))';
            end
            
            for i = 1:m
                if size(Obj.Controller.CodeGivenState, 3) == 1
                    bayes_transitions(:, :, i, t) = Obj.Controller.CodeGivenState * Obj.Transitions(:, :, i) * state_given_code(:, :, t);
                else
                    bayes_transitions(:, :, i, t) = Obj.Controller.CodeGivenState(:, :, t + 1) * Obj.Transitions(:, :, i) * state_given_code(:, :, t);
                end
            end
        end
    else
        error('Not implemented');
    end
end

if isequal(Obj.SolverName, 'Info') && isequal('Code', Obj.Controller.Estimator)
    belief_dist = zeros(p, horizon + 1);
    proc_dist = [code_dist(:, 1) zeros(p, horizon)];
else
    belief_dist = zeros(n, horizon + 1);
    proc_dist = [init_dist zeros(n, horizon)];
end

traj_dist = init_dist;

for t = 1:horizon
    output_dist = output_given_state(:, traj(t));
    output = discretesample(output_dist, 1);
    
    if isequal(Obj.SolverName, 'Info') && isequal('Code', Obj.Controller.Estimator)
        output_given_code = output_given_state * state_given_code(:, :, t);
        belief_dist(:, t) = output_given_code(output, :)' .* proc_dist(:, t);
        belief_dist(:, t) = belief_dist(:, t) ./ sum(belief_dist(:, t));
        
        [~, mle] = max(belief_dist(:, t));
        input_dist = Obj.Controller.InputGivenCode(:, mle, t);
    else
        belief_dist(:, t) = output_given_state(output, :)' .* proc_dist(:, t);
        belief_dist(:, t) = belief_dist(:, t) ./ sum(belief_dist(:, t));
        
        [~, mle] = max(belief_dist(:, t));
        input_dist = Obj.Controller.Policy(:, mle, t);
    end
    
    input = discretesample(input_dist, 1);
    costs(t) = Obj.Costs(traj(t), input);
    traj_dist = Obj.Transitions(:, :, input) * traj_dist;
    traj(t + 1) = discretesample(traj_dist, 1);
    
    if isequal(Obj.SolverName, 'Info') && isequal('Code', Obj.Controller.Estimator)
    else 
        proc_dist(:, t + 1) = Obj.Transitions(:, :, input) * belief_dist(:, t);
    end
end

costs(end) = Obj.TerminalCosts(traj(end));
cum_cost = cumsum(costs);

end

function T = forward_eqn(transitions, policy)
    T = zeros(size(policy, 2));
    
    for i = 1:size(policy, 2)
        T(:, i) = squeeze(transitions(:, i, :)) * policy(:, i);
    end
end

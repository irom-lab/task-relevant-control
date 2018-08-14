function [ traj, cum_cost ] = sim_init_uncertainty(Obj, init_dist, horizon)
%SIMULATE_INIT_UNCERTAINTY Summary of this function goes here
%   Detailed explanation goes here

if isinf(Obj.Parameters.Horizon) && nargin < 3
    error('Control problem is infinite horizon. Specify horizon for sim.');
elseif isfinite(Obj.Parameters.Horizon)
    horizon = Obj.Parameters.Horizon;
end

traj = [discretesample(init_dist, 1); zeros(horizon, 1)];
costs = zeros(horizon + 1, 1);
state_dist = init_dist;

for t = 1:horizon
    [~, mle] = max(state_dist);
    
    if isfinite(Obj.Parameters.Horizon)
        input_dist = Obj.Controller(:, mle, t);
    else
        input_dist = Obj.Controller(:, mle);
    end
    
    input = discretesample(input_dist, 1);
    costs(t) = Obj.Costs(traj(t), input);
    state_dist = Obj.Transitions(:, :, input) * state_dist;
    traj(t + 1) = discretesample(state_dist, 1);
end

costs(end) = Obj.TerminalCosts(traj(end));
cum_cost = cumsum(costs);

end


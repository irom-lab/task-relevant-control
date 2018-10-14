function [full_traj] = full_slip_traj(ic, inputs, Parameters)
%FULL_SLIP_TRAJ Summary of this function goes here
%   Detailed explanation goes here

full_traj = [];
state = ic;

for input = inputs
    [state, traj] = slip_return_map(state, input, Parameters, true);
    
    full_traj = [full_traj traj];
end

end


function [ traj, cum_cost ] = sim_meas_uncertainty(Obj, horizon)
%SIM_MEAS_UNCERTAINTY Summary of this function goes here
%   Detailed explanation goes here

traj = [mvnrnd(Obj.Init.mean, Obj.Init.cov)' zeros(Obj.Parameters.NStates, horizon)];
costs = zeros(horizon + 1, 1);

meas_cov = Obj.Parameters.MeasCov;
actual_meas_cov = 10 * rand(size(meas_cov, 1));
actual_meas_cov = actual_meas_cov' * actual_meas_cov;

kalman_est = Obj.Init.mean;
kalman_cov = Obj.Init.cov;

for t = 1:horizon
    measurement = mvnrnd(traj(:, t), actual_meas_cov)';
    
    kalman_est = kalman_est + kalman_cov * inv(kalman_cov + meas_cov) * (measurement - kalman_est);
    kalman_cov =  kalman_cov - kalman_cov * inv(kalman_cov + meas_cov) * kalman_cov';
    
    if isequal(Obj.SolverName, 'Exact')
        input = Obj.Controller.K(:, :, t) * kalman_est + Obj.Controller.d(:, t);
    elseif isequal(Obj.SolverName, 'Info')
        input = Obj.Controller.K(:, :, t) * (Obj.Controller.C(:, :, t) * kalman_est ...
            + Obj.Controller.d(:, t) + mvnrnd(zeros(Obj.Parameters.NInputs, 1), Obj.Controller.Sigma_eta(:, :, t))');
    end
    
    costs(t) = cost(Obj, traj(:, t), input, t);
    traj(:, t + 1) = dynamics(Obj, traj(:, t), input) + mvnrnd(zeros(Obj.Parameters.NStates, 1), Obj.Parameters.ProcCov)';
    
    [A, B] = linearize(Obj, kalman_est, input);
    kalman_est = A * kalman_est + B * input;
    kalman_cov = A * kalman_cov * A' + Obj.Parameters.ProcCov;
end

costs(end) = terminal_cost(Obj, traj(:, end));

cum_cost = cumsum(costs);

end


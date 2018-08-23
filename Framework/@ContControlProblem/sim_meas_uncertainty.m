function [ traj, cum_cost ] = sim_meas_uncertainty(Obj, init, horizon)
%SIM_MEAS_UNCERTAINTY Summary of this function goes here
%   Detailed explanation goes here

traj = [mvnrnd(init.mean, init.cov)' zeros(Obj.Parameters.NStates, horizon)];
costs = zeros(horizon + 1, 1);

meas_cov = Obj.Parameters.MeasCov;

kalman_est = init.mean;
kalman_cov = init.cov;

for t = 1:horizon
    measurement = mvnrnd(traj(:, t), meas_cov)';
    
    kalman_est = kalman_est + kalman_cov * inv(kalman_cov + meas_cov) * (measurement - kalman_est);
    kalman_cov =  kalman_cov - kalman_cov * inv(kalman_cov + meas_cov) * kalman_cov';
    
    if isequal(Obj.SolverName, 'Exact')
        input = Obj.Controller(:, :, t) * kalman_est;
    end
    
    costs(t) = cost(Obj, traj(:, t), input);
    traj(:, t + 1) = dynamics(Obj, traj(:, t), input) + mvnrnd(zeros(Obj.Parameters.NStates, 1), Obj.Parameters.ProcCov)';
    
    [A, B] = linearize(Obj, kalman_est, input);
    kalman_est = A * kalman_est;
    kalman_cov = A * kalman_cov * A' + Obj.Parameters.ProcCov;
end

cum_cost = cumsum(costs);

end


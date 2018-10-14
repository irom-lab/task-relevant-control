function [ traj, inputs, cum_cost ] = sim_meas_uncertainty(Obj, horizon, actual_meas_cov)
%SIM_MEAS_UNCERTAINTY Summary of this function goes here
%   Detailed explanation goes here

traj = [mvnrnd(zeros(Obj.Parameters.NStates, 1), Obj.Init.cov)' zeros(Obj.Parameters.NStates, horizon)];
inputs = zeros(Obj.Parameters.NInputs, horizon);
costs = zeros(horizon + 1, 1);

meas_cov = Obj.Parameters.MeasCov;

if nargin < 3
    actual_meas_cov = rand(size(meas_cov, 1));
    actual_meas_cov = actual_meas_cov' * actual_meas_cov;
end

% kalman_proc_est = zeros(Obj.Parameters.NStates, 1);
% kalman_proc_cov = Obj.Init.cov;

if isequal(Obj.SolverName, 'Exact')
    kalman_proc_est = zeros(Obj.Parameters.NStates, 1);
    kalman_proc_cov = Obj.Init.cov;
elseif isequal(Obj.SolverName, 'Info')
    kalman_proc_est = Obj.Controller.d(:, 1);
    kalman_proc_cov = Obj.Controller.C(:, :, 1) * Obj.Init.cov * Obj.Controller.C(:, :, 1)' + Obj.Controller.Sigma_eta(:, :, 1);
end

for t = 1:horizon
    measurement = mvnrnd(traj(:, t), actual_meas_cov)';        
    
    A = Obj.Controller.A(:, :, t);
    B = Obj.Controller.B(:, :, t);
    D = eye(4);
    
    if isequal(Obj.SolverName, 'Exact')
                
        [kalman_meas_est, kalman_meas_cov] = kalman_meas_update(D, meas_cov, kalman_proc_cov, kalman_proc_est, measurement);
        
        inputs(:, t) = Obj.Controller.K(:, :, t) * kalman_meas_est + Obj.Controller.f(:, t);                
        
        [kalman_proc_est, kalman_proc_cov] = kalman_proc_update(A, B, Obj.Parameters.ProcCov, kalman_meas_est, kalman_meas_cov, inputs(:, t));
        
    elseif isequal(Obj.SolverName, 'Info')
        C = Obj.Controller.C(:, :, t);
        d = Obj.Controller.d(:, t);
        Sigma_eta = Obj.Controller.Sigma_eta(:, :, t);
        Sigma_epsilon = Obj.Parameters.ProcCov;
        Sigma_omega = meas_cov;
        x_bar = Obj.Controller.x_bar(:, t);
        Sigma_x = Obj.Controller.Sigma_x(:, :, t);
        x_tilde_bar = C * x_bar + d;
        Sigma_x_tilde = C * Sigma_x * C' + Sigma_eta;

        
%         [kalman_meas_est, kalman_meas_cov] = kalman_meas_update(D, meas_cov, kalman_proc_cov, kalman_proc_est, measurement);
%         
%         inputs(:, t) = Obj.Controller.K(:, :, t) * (C * kalman_meas_est ...
%             + d + 0 * mvnrnd(zeros(size(C, 1), 1), Sigma_eta)') + Obj.Controller.f(:, t);
%         
%         [kalman_proc_est, kalman_proc_cov] = kalman_proc_update(A, B, Obj.Parameters.ProcCov, kalman_meas_est, kalman_meas_cov, inputs(:, t));                  
        
        
        [C_tilde, Sigma_omega_tilde] = trv_measurement_variables(C, D, Sigma_omega, Sigma_x, Sigma_x_tilde);
        
        z_measurement = measurement - D * x_bar + C_tilde * x_tilde_bar;
        
        [kalman_meas_est, kalman_meas_cov] = kalman_meas_update(C_tilde, Sigma_omega_tilde, kalman_proc_cov, kalman_proc_est, z_measurement);
        
        
        inputs(:, t) = Obj.Controller.K(:, :, t) * kalman_meas_est + Obj.Controller.f(:, t);
        
        if t < horizon
            C_next = Obj.Controller.C(:, :, t + 1);
            d_next = Obj.Controller.d(:, t + 1);
            Sigma_eta_next = Obj.Controller.Sigma_eta(:, :, t + 1);
        
            
            [A_tilde, B_tilde, r_tilde, Sigma_epsilon_tilde] = trv_process_variables(A, B, C, C_next, d_next, Sigma_eta_next, Sigma_epsilon, x_bar, Sigma_x, x_tilde_bar, Sigma_x_tilde);
            
            [kalman_proc_est, kalman_proc_cov] = kalman_proc_update(A_tilde, B_tilde, Sigma_epsilon_tilde, kalman_meas_est, kalman_meas_cov, inputs(:, t));

            kalman_proc_est = kalman_proc_est + r_tilde;
        end
    end
    
    costs(t) = cost(Obj, traj(:, t) + Obj.Controller.nominal_states(:, t), inputs(:, t) + Obj.Controller.nominal_inputs(:, t), t);
    next_state = slip_return_map(traj(:, t) + Obj.Controller.nominal_states(:, t), ...
        Obj.Controller.nominal_inputs(:, t) + inputs(:, t), Obj.Parameters, true);
    
    if any(isnan(next_state)) || any(isnan(A(:)))
        cum_cost = nan;
        return;
    end  
    
    traj(:, t + 1) = next_state - Obj.Controller.nominal_states(:, t + 1) + mvnrnd(zeros(Obj.Parameters.NStates, 1), Obj.Parameters.ProcCov)';        
end

costs(end) = terminal_cost(Obj, Obj.Controller.nominal_states(:, end) + traj(:, end));
traj = traj + Obj.Controller.nominal_states;
inputs = inputs + Obj.Controller.nominal_inputs;
cum_cost = cumsum(costs);

end

function [kalman_proc_est, kalman_proc_cov] = kalman_proc_update(A, B, proc_cov, kalman_meas_est, kalman_meas_cov, input)
    kalman_proc_est = A * kalman_meas_est + B * input;
    kalman_proc_cov = A * kalman_meas_cov * A' + proc_cov;
end

function [kalman_meas_est, kalman_meas_cov] = kalman_meas_update(D, meas_cov, kalman_proc_cov, kalman_proc_est, measurement)
    kalman_meas_est = kalman_proc_est + kalman_proc_cov * D' * inv(D * kalman_proc_cov * D' + meas_cov) * (measurement - D * kalman_proc_est);
    kalman_meas_cov =  kalman_proc_cov - kalman_proc_cov * D' * inv(D * kalman_proc_cov * D' + meas_cov) * D * kalman_proc_cov;
end

function [A_tilde, B_tilde, r_tilde, Sigma_epsilon_tilde] = trv_process_variables(A, B, C, C_next, d_next, Sigma_eta_next, Sigma_epsilon, x_bar, Sigma_x, x_tilde_bar, Sigma_x_tilde)
    A_tilde = C_next * A * Sigma_x * C' * inv(Sigma_x_tilde);
    
    B_tilde = C_next * B;
    
    r_tilde = -A_tilde * x_tilde_bar + C_next * A * x_bar + d_next;    
    
    Sigma_epsilon_tilde = C_next * A * (Sigma_x - Sigma_x * C' * inv(Sigma_x_tilde) * C * Sigma_x) * A' * C_next' + C_next * Sigma_epsilon * C_next' + Sigma_eta_next;        
end

function [C_tilde, Sigma_omega_tilde] = trv_measurement_variables(C, D, Sigma_omega, Sigma_x, Sigma_x_tilde)
    C_tilde = D * Sigma_x * C' * inv(Sigma_x_tilde);
    
    Sigma_omega_tilde = D * Sigma_x * D' - D * Sigma_x * C' * inv(Sigma_x_tilde) * C * Sigma_x * D' + Sigma_omega;
end








close all;
clear;
clc;

%%

n = 2;
m = 2;

Init = 5 * rand(n, 1);

Parameters.A = rand(n, n);
Parameters.B = rand(n, m);
Parameters.Q = diag([1 1]);
Parameters.R = eye(m);
Parameters.MeasCov = diag([0.1 0.1]);
Parameters.ProcCov = zeros(2);

Parameters.NStates = 2;
Parameters.NInputs = 2;

Parameters.Horizon = 30;

SolverOptions.Iters = 5;
Problem = LTIProblem(Parameters, SolverOptions, Init);

%%

[controller, obj_val, obj_hist] = solve_exact(Problem);

%%

init.mean = [5; 3];
init.cov = 0.1 * eye(2);

[traj, costs] = sim_meas_uncertainty(Problem, init, Parameters.Horizon);

figure;
hold on;

plot(traj(1, :), traj(2, :), 'LineWidth', 2);

xlim([-5, 5]);
ylim([-5, 5]);
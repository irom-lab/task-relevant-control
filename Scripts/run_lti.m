close all;
clear;
clc;

%%

n = 2;
m = 2;

Init = 5 * rand(n, 1);

Parameters.A = rand(n, n);
Parameters.B = rand(n, m);
Parameters.Q = diag([100 0]);
Parameters.R = eye(m);

Parameters.Horizon = 5;

SolverOptions.Iters = 5;
Problem = LTIProblem(Parameters, SolverOptions, Init);

%%

[controller, obj_val, obj_hist] = solve_exact(Problem);

%%

[traj, costs] = simulate(Problem);

figure;
hold on;

plot(traj(1, :), traj(2, :), 'LineWidth', 2);

xlim([-5, 5]);
ylim([-5, 5]);
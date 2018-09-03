close all;
clear;
clc;

load slip_steady_state.mat

%%
rng(0);

n = 2;
m = 2;

Init.mean = [0; slip_steady_state];
Init.cov = 0.5 * eye(4);

Parameters.MeasCov = diag([0.1 0.1]);
Parameters.ProcCov = 0 * diag([0.1, 0.1]);

Parameters.Horizon = 3;
Parameters.Goals = [0 0 0 3.3; zeros(4, 4)];

Parameters.NomInputs = slip_steady_state(2) * ones(1, Parameters.Horizon);
Parameters.Q = zeros(5, 5, Parameters.Horizon + 1);
Parameters.Q(1, 1, end) = 100;
Parameters.R = ones(1, 1, Parameters.Horizon);

Parameters.Delta = 1e-6;

SolverOptions.Iters = 5;

%%

Problem = SLIPProblem(Parameters, SolverOptions, Init);

%%

[controller, obj_val, obj_hist] = solve_ilqr(Problem);
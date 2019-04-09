clear;
close all;
clc;

rng(0);

%%

Init.mean = [0; 4; 4; 0; -1.5; 0; 1];
Init.cov = diag([1e-1 1e-1 0 0 1e-1 0 0]);

Parameters.Gravity = 9.8;
Parameters.f = 10;
Parameters.CenterY = 0;
Parameters.DeltaT = 1/60;

flight_time = (Init.mean(6) + sqrt(Init.mean(6) ^ 2 + 2 * Parameters.Gravity * Init.mean(3))) / Parameters.Gravity;

Parameters.Horizon = floor(flight_time / Parameters.DeltaT);

Parameters.Q = zeros(7, 7, Parameters.Horizon + 1);
Parameters.Q(:, :, end) = 100 * [1; -1; 0; 0; 0; 0; 0] * [1 -1 0 0 0 0 0];
Parameters.R = 0.1 * ones(1, 1, Parameters.Horizon);
Parameters.Goals = zeros(7, Parameters.Horizon + 1);
Parameters.NomInputs = zeros(1, Parameters.Horizon);

Parameters.ProcCov = 1e-3 * diag([1 1 1 1 1 1 0]);

SolverOptions.Tradeoff = 1;
SolverOptions.NumCodewords = 1;
SolverOptions.FixedCodeMap = false;
SolverOptions.Iters = 5;

Problem = GazeProblem(Parameters, SolverOptions, Init);

%%

[controller, obj_val, obj_hist] = solve_ilqr(Problem);

%%


%Problem.SolverOptions.InitController = controller;
Problem.Parameters.NomInputs = controller.nominal_inputs;

[controller, obj_val, obj_hist] = solve_info_ilqr(Problem);

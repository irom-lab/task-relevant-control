clc;
close all;
clear;

%%

rng(0);

Parameters.Length = 4;
Parameters.Goal = 3;
Parameters.Horizon = 5;
Parameters.MeasurementRate = 0.5;

SolverOptions.Tradeoff = 20;
SolverOptions.Iters = 100;
SolverOptions.NumCodewords = 5;
SolverOptions.FixedCodeMap = false;

init_dist = [0.3; 0.4; 0; 0.3; 0];
init_dist = init_dist ./ sum(init_dist);

Problem = LavaProblem(Parameters, SolverOptions, init_dist);

%%

controller = solve_exact(Problem);

%%

Problem.SolverOptions.InitPolicy.CodeGivenState = repmat(eye(5), 1, 1, 5);
Problem.SolverOptions.InitPolicy.InputGivenCode = controller.Policy;

tic;
solve_info(Problem)
toc
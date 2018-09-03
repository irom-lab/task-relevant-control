close all;
clear;
clc;

load slip_steady_state.mat

%%
rng(0);

n = 2;
m = 2;

Init.mean = [5; 3];
Init.cov = 0.5 * eye(2);

Parameters.A = eye(2);%rand(n, n);
Parameters.B = eye(2);%rand(n, m);
Parameters.Q = diag([100 100]);
Parameters.R = eye(m);
Parameters.MeasCov = diag([0.1 0.1]);
Parameters.ProcCov = diag([0.1, 0.1]);

Parameters.NStates = 5;
Parameters.NInputs = 1;

Parameters.Horizon = 3;
Parameters.Goals = [0 0 0 3; zeros(5, 4)];
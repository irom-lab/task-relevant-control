%close all;
clear;
clc;

load slip_steady_state.mat

%%


actual_meas_cov_scale = 1e-3;
believed_meas_cov_scale = 1e-4;

Init.mean = [0.01; -0.01; 0.01];
Init.cov = 0.001 * eye(3);
Parameters.MeasCov = believed_meas_cov_scale * eye(3);
Parameters.ProcCov = 0.01 * diag([0.01; 0.05; 0.05]);
Parameters.Horizon = 5;
Parameters.Goals = zeros(3, Parameters.Horizon + 1);
Parameters.NomInputs = zeros(1, Parameters.Horizon);

% Linearize
A = zeros(3);
B = 0;

for i = 1:3
    x = slip_steady_state(2:end);
    x(i) = x(i) + 1e-6;
    forward = slip_return_map([0; x], 0, Parameters, false);

    x = slip_steady_state(2:end);
    x(i) = x(i) - 1e-6;
    reverse = slip_return_map([0; x], 0, Parameters, false);

    A(:, i) = (forward(2:end) - reverse(2:end)) / (2 * 1e-6);
end

forward = slip_return_map([0; slip_steady_state(2:end)], 1e-6, Parameters, false);
reverse = slip_return_map([0; slip_steady_state(2:end)], -1e-6, Parameters, false);
B = (forward(2:end) - reverse(2:end)) / (2 * 1e-6);

Parameters.A = A;
Parameters.B = B;
Parameters.Q = eye(3);
Parameters.R = 100;
Parameters.NStates = 3;
Parameters.NInputs = 1;

SolverOptions.Tradeoff = 10;
SolverOptions.NumCodewords = 3;
SolverOptions.FixedCodeMap = false;
SolverOptions.Iters = 12;

samples = 500;

Problem = LTIProblem(Parameters, SolverOptions, Init);

%%

[controller, obj_val, obj_hist] = solve_exact(Problem);

%%
fprintf('Running N = %d simulations...\n', samples);

rng(0);

c1 = zeros(Parameters.Horizon + 1, samples);
trajs = zeros(3, Parameters.Horizon + 1, samples);
successes1 = false(1, samples);

tic;
for i = 1:samples
    actual_meas_cov = rand(size(Parameters.MeasCov, 1));
    actual_meas_cov =  actual_meas_cov_scale * (actual_meas_cov' * actual_meas_cov);
    
    
    [trajs(:, :, i), c1(:, i)] = sim_slip_uncertainty(Problem, slip_steady_state(2:4), Parameters.Horizon, actual_meas_cov);
    successes1(i) = all(~isnan(c1(:, i)));
end

time = toc;



%%

fprintf('Running N = %d simulations...\n', samples);

rng(0);

c2 = zeros(Parameters.Horizon + 1, samples);
successes2 = false(1, samples);

tic;
for i = 1:samples
    actual_meas_cov = rand(size(Parameters.MeasCov, 1));
    actual_meas_cov = actual_meas_cov_scale * (actual_meas_cov' * actual_meas_cov);
    Problem.Parameters.MeasCov = actual_meas_cov;
    
    [trajs(:, :, i), c2(:, i)] = sim_slip_uncertainty(Problem, slip_steady_state(2:4), Parameters.Horizon, Problem.Parameters.MeasCov);
    successes2(i) = all(~isnan(c2(:, i)));
end

time = toc;


%%

% rng(0);
% [controller, obj_val, obj_hist, mean_traj, mean_inputs] = solve_info(Problem);
% fprintf('\n');

%%

rng(0);
[controller, obj_val, obj_hist] = solve_info_lqg(Problem);
fprintf('\n');

%%

% rng(0);
% [controller, obj_val, obj_hist] = solve_info_ilqr(Problem);
% fprintf('\n');

%%

fprintf('Running N = %d simulations...\n', samples);

rng(0);

c3 = zeros(Parameters.Horizon + 1, samples);
trajs = zeros(3, Parameters.Horizon + 1, samples);
successes3 = false(1, samples);
Problem.Parameters.MeasCov = believed_meas_cov_scale * eye(3);

tic;
for i = 1:samples
    actual_meas_cov =  rand(size(Parameters.MeasCov, 1));
    actual_meas_cov =  actual_meas_cov_scale * (actual_meas_cov' * actual_meas_cov);
    
    [trajs(:, :, i), c3(:, i)] = sim_slip_uncertainty(Problem, slip_steady_state(2:4), Parameters.Horizon, actual_meas_cov);
    successes3(i) = all(~isnan(c3(:, i)));
end

time = toc;

%%

fig = figure;
hold on;

title(['\beta = ' num2str(SolverOptions.Tradeoff)]);
xlabel('Timestep', 'FontSize', 15);
ylabel('Avg. Cummulative Cost', 'FontSize', 15);
xlim([1 Parameters.Horizon + 1]);

h1 = plot(1:size(c1(:, successes1), 1), mean(c1(:, successes1), 2), '-k');
h2 = plot(1:size(c2(:, successes2), 1), mean(c2(:, successes2), 2), '-b');
h3 = plot(1:size(c3(:, successes3), 1), mean(c3(:, successes3), 2), '-r');

shadedErrorBar(1:size(c1(:, successes1), 1), mean(c1(:, successes1), 2), std(c1(:, successes1), 0, 2), 'lineprops', '-k');
shadedErrorBar(1:size(c2(:, successes2), 1), mean(c2(:, successes2), 2), std(c2(:, successes2), 0, 2), 'lineprops', '-b');
shadedErrorBar(1:size(c3(:, successes3), 1), mean(c3(:, successes3), 2), std(c3(:, successes3), 0, 2), 'lineprops', '-r');


plot(1:size(c1(:, successes1), 1), mean(c1(:, successes1), 2), '-k', 'LineWidth', 2);
plot(1:size(c2(:, successes2), 1), mean(c2(:, successes2), 2), '-b', 'LineWidth', 2);
plot(1:size(c3(:, successes3), 1), mean(c3(:, successes3), 2), '-r', 'LineWidth', 2);

scatter(1:size(c1(:, successes1), 1), mean(c1(:, successes1), 2), 'k', 'filled');
scatter(1:size(c2(:, successes2), 1), mean(c2(:, successes2), 2), 'b', 'filled');
scatter(1:size(c3(:, successes3), 1), mean(c3(:, successes3), 2), 'r', 'filled');

legend([h1 h2 h3], {'Incorrect Covariance LQG', 'Correct Covariance LQG', 'Incorrect Covariance TRV'});

set(gcf,'Units','inches');
screenposition = get(gcf,'Position');
set(gcf,...
     'PaperPosition',[0 0 screenposition(3:4)],...
     'PaperSize',[screenposition(3:4)]);
 print -dpdf -painters Figures/lqg_result.pdf
 savefig(fig, 'Figures/lqg_result.fig', 'compact');
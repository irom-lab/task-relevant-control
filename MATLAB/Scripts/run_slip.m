%close all;
clear;
clc;

load slip_steady_state.mat

%%
rng(0);

n = 2;
m = 2;


actual_meas_cov_scale = 1e-3;
believed_meas_cov_scale = 1e-4;

Init.mean = [0; slip_steady_state([2:4])];
Init.cov = 1e-3 * eye(4);
Parameters.MeasCov = believed_meas_cov_scale * eye(4);
Parameters.ProcCov = 0.0001 * diag([1; 0.1; 0.5; 0.5]);
Parameters.Horizon = 3;
Parameters.Goals = [0 0 0 3.2; zeros(3, 4)];
Parameters.TMax = 2;
Parameters.NomInputs = slip_steady_state(2) * zeros(1, Parameters.Horizon);
Parameters.Q = zeros(4, 4, Parameters.Horizon + 1);
Parameters.Q(1, 1, end) = 1;
Parameters.R = 10 * ones(1, 1, Parameters.Horizon);

Parameters.Delta = 1e-6;
SolverOptions.Tradeoff = 100;
SolverOptions.NumCodewords = 4;
SolverOptions.FixedCodeMap = false;
SolverOptions.Iters = 10;

samples = 500;

betas_bounds = [200 1];

Problem = SLIPProblem(Parameters, SolverOptions, Init);

%%

[controller, obj_val, obj_hist] = solve_ilqr(Problem);

%%

fprintf('Running N = %d simulations...\n', samples);

rng(0);

c1 = zeros(Parameters.Horizon + 1, samples);
successes1 = false(1, samples);

inputs1 = zeros(1, Parameters.Horizon, samples);

tic;

for i = 1:samples
    actual_meas_cov =  rand(size(Parameters.MeasCov, 1));
    actual_meas_cov =  actual_meas_cov_scale * actual_meas_cov' * actual_meas_cov;
    
    [trajs1(:, :, i), inputs1(:, :, i), c] = sim_meas_uncertainty(Problem, Parameters.Horizon, actual_meas_cov);
    
    if ~isnan(c)
        c1(:, i) = c;
        successes1(i) = true;
    end
end

time = toc;

fprintf('\tFinished in %fs\n', time);

%%

fprintf('Running N = %d simulations...\n', samples);

rng(0);

c2 = zeros(Parameters.Horizon + 1, samples);
successes2 = false(1, samples);
inputs2 = zeros(1, Parameters.Horizon, samples);
tic;

for i = 1:samples
    actual_meas_cov =  rand(size(Parameters.MeasCov, 1));
    actual_meas_cov =  actual_meas_cov_scale * actual_meas_cov' * actual_meas_cov;
    Problem.Parameters.MeasCov = actual_meas_cov;
    
    [trajs2(:, :, i), inputs2(:, :, i), c] = sim_meas_uncertainty(Problem, Parameters.Horizon, Problem.Parameters.MeasCov);
    
    if ~isnan(c)
        c2(:, i) = c;
        successes2(i) = true;
    end
end

time = toc;

fprintf('\tFinished in %fs\n', time);


%%

betas = linspace(betas_bounds(1), betas_bounds(2), 10);

BestProblem = [];
best_mi = Inf;

for i = 1:length(betas)
    fprintf('Iter %d of %d\tBeta = %f\n', i, length(betas), betas(i));
    rng(0);
    
    Problem.SolverOptions.Tradeoff = betas(i);
    
    [controller, obj_val, obj_hist, expected_cost, mi, ~, ~] = solve_info_ilqr(Problem);
    
    if expected_cost < 0.5 && mi < best_mi
        BestProblem = copy(Problem);
        best_mi = mi;
        best_beta = betas(i);
    end
end

Problem = copy(BestProblem);


%%


fprintf('Running N = %d simulations...\n', samples);

rng(0);

Problem.Parameters.MeasCov = believed_meas_cov_scale * eye(4);

c3 = zeros(Parameters.Horizon + 1, samples);
successes3 = false(1, samples);
inputs3 = zeros(1, Parameters.Horizon, samples);
tic;

for i = 1:samples
    actual_meas_cov = rand(size(Parameters.MeasCov, 1));
    actual_meas_cov = actual_meas_cov_scale * actual_meas_cov' * actual_meas_cov;
    
    [trajs3(:, :, i), inputs3(:, :, i), c] = sim_meas_uncertainty(Problem, Parameters.Horizon, actual_meas_cov);
    
    if ~isnan(c)
        c3(:, i) = c;
        successes3(i) = true;
    end
end

time = toc;

fprintf('\tFinished in %fs\n', time);

%%
close all;
figure;
hold on;

histogram(c2(end, successes2), linspace(0, 0.5, 50), 'FaceColor', 'g', 'FaceAlpha', 1); % Incorrect iLQG Cov
histogram(c3(end, successes3), 0.005 + linspace(0, 0.5, 50), 'FaceColor', 'b', 'FaceAlpha', 0.8); % Incorrect TRV Cov
histogram(c1(end, successes1), 0.0075 + linspace(0, 0.5, 50), 'FaceColor', 'r', 'FaceAlpha', 0.8); % Correct iLQG Cov

% title(['\beta = ' num2str(SolverOptions.Tradeoff)]);
% xlabel('Timestep', 'FontSize', 15);
% ylabel('Cummulative Cost', 'FontSize', 15);
% xlim([1 Parameters.Horizon + 1]);
% 
% h1 = plot(1:size(c1(:, successes1), 1), mean(c1(:, successes1), 2), '-k');
% h2 = plot(1:size(c2(:, successes2), 1), mean(c2(:, successes2), 2), '-b');
% h3 = plot(1:size(c3(:, successes3), 1), mean(c3(:, successes3), 2), '-r');
% 
% shadedErrorBar(1:size(c1(:, successes1), 1), mean(c1(:, successes1), 2), std(c1(:, successes1), 0, 2), 'lineprops', '-k');
% shadedErrorBar(1:size(c2(:, successes2), 1), mean(c2(:, successes2), 2), std(c2(:, successes2), 0, 2), 'lineprops', '-b');
% shadedErrorBar(1:size(c3(:, successes3), 1), mean(c3(:, successes3), 2), std(c3(:, successes3), 0, 2), 'lineprops', '-r');
% 
% 
% plot(1:size(c1(:, successes1), 1), mean(c1(:, successes1), 2), '-k', 'LineWidth', 2);
% plot(1:size(c2(:, successes2), 1), mean(c2(:, successes2), 2), '-b', 'LineWidth', 2);
% plot(1:size(c3(:, successes3), 1), mean(c3(:, successes3), 2), '-r', 'LineWidth', 2);
% 
% scatter(1:size(c1(:, successes1), 1), mean(c1(:, successes1), 2), 'k', 'filled');
% scatter(1:size(c2(:, successes2), 1), mean(c2(:, successes2), 2), 'b', 'filled');
% scatter(1:size(c3(:, successes3), 1), mean(c3(:, successes3), 2), 'r', 'filled');
% 
% legend([h1 h2 h3], {'Incorrect Covariance ILQG', 'Correct Covariance ILQG', 'Incorrect Covariance TRV'});
% 
% set(gcf,'Units','inches');
% screenposition = get(gcf,'Position');
% set(gcf,...
%      'PaperPosition',[0 0 screenposition(3:4)],...
%      'PaperSize',[screenposition(3:4)]);
%  print -dpdf -painters fig
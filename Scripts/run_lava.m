clear;
close all;
clc;

%%

rng(0);

samples = 50000;

Parameters.Length = 4;
Parameters.Goal = 3;
Parameters.Horizon = 10;

SolverOptions.Tradeoff = 0.01;
SolverOptions.Iters = 30;
SolverOptions.NumCodewords = 3;
SolverOptions.FixedCodeMap = true;

meas_rate = 0.4;

init_dist = [0.3; 0.4; 0; 0.3; 0];
init_dist = init_dist ./ sum(init_dist);

Problem = LavaProblem(Parameters, SolverOptions, init_dist);

%%

disp('Solving problem exactly...');

tic;
[controller, objective] = solve_exact(Problem);
time = toc;

fprintf('\tObjective: %f\n', objective);
fprintf('\tFinished in %fs\n', time);

%%

fprintf('Running N = %d simulations...\n', samples);

rng(0);

c1 = zeros(Parameters.Horizon + 1, samples);

tic;
parfor i = 1:samples
    [~, c1(:, i)] = sim_meas_uncertainty(Problem, init_dist, meas_rate);
end

time = toc;

fprintf('\tFinished in %fs\n', time);

%%

%%

disp('Solving problem with information constraint...');

tic;
[~, objective, obj_hist] = solve_info(Problem);
time = toc;

fprintf('\tObjective: %f\n', objective);
fprintf('\tFinished in %fs\n', time);

%%

rng(0);

c2 = zeros(Parameters.Horizon + 1, samples);

fprintf('Running N = %d simulations w/ state estimator...\n', samples);

tic;
parfor i = 1:samples
    [~, c2(:, i)] = sim_meas_uncertainty(Problem,  init_dist, meas_rate);
end
time = toc;

fprintf('\tFinished in %fs\n', time);


%%

%%

rng(0);

c3 = zeros(Parameters.Horizon + 1, samples);
Problem.Controller.Estimator = 'Code';

fprintf('Running N = %d simulations w/ code estimator...\n', samples);

tic;
parfor i = 1:samples
    [~, c3(:, i)] = sim_meas_uncertainty(Problem,  init_dist, meas_rate);
end
time = toc;

fprintf('\tFinished in %fs\n', time);

%%


figure;
hold on;

title(['\beta = ' num2str(SolverOptions.Tradeoff)]);
xlabel('Timestep');
ylabel('Cummulative Cost');
xlim([1 Parameters.Horizon + 1]);

plot(1:size(c1, 1), mean(c1, 2), '-k');
plot(1:size(c2, 1), mean(c2, 2), '-b');
plot(1:size(c3, 1), mean(c3, 2), '-r');

legend('Exact', 'Info State', 'Info Code');

shadedErrorBar(1:size(c1, 1), mean(c1, 2), std(c1, 0, 2), 'lineprops', '-k');
shadedErrorBar(1:size(c2, 1), mean(c2, 2), std(c2, 0, 2), 'lineprops', '-b');
shadedErrorBar(1:size(c3, 1), mean(c3, 2), std(c3, 0, 2), 'lineprops', '-r');

plot(1:size(c1, 1), mean(c1, 2), '-k', 'LineWidth', 2);
plot(1:size(c2, 1), mean(c2, 2), '-b', 'LineWidth', 2);
plot(1:size(c3, 1), mean(c3, 2), '-r', 'LineWidth', 2);

scatter(1:size(c1, 1), mean(c1, 2), 'k', 'filled');
scatter(1:size(c2, 1), mean(c2, 2), 'b', 'filled');
scatter(1:size(c3, 1), mean(c3, 2), 'r', 'filled');

%%

figure;
hold on;
plot(obj_hist, '-k', 'LineWidth', 2);
scatter(1:length(obj_hist), obj_hist, 'k', 'filled');
xlabel('Iter');
ylabel('Objective');
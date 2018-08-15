clear;
close all;
clc;

%%

rng(0);

samples = 1000;

Parameters.Runway = 10;
Parameters.Height = 5;
Parameters.Horizon = 5;
Parameters.MeasurementRate = 0.4;

SolverOptions.Tradeoff = 1;
SolverOptions.Iters = 30;
SolverOptions.NumCodewords = 100;
SolverOptions.FixedCodeMap = true;

init_dist = zeros(Parameters.Runway * Parameters.Height * Parameters.Runway * 2, 1);
init_dist(sub2ind([Parameters.Runway Parameters.Runway Parameters.Height 2], 1, Parameters.Runway, Parameters.Height, 1)) = 1;
init_dist(sub2ind([Parameters.Runway Parameters.Runway Parameters.Height 2], 1, Parameters.Runway, Parameters.Height, 2)) = 1;
init_dist(sub2ind([Parameters.Runway Parameters.Runway Parameters.Height 2], 1, Parameters.Runway, Parameters.Height - 1, 1)) = 1;
init_dist(sub2ind([Parameters.Runway Parameters.Runway Parameters.Height 2], 1, Parameters.Runway, Parameters.Height - 1, 2)) = 1;

init_dist = init_dist ./ sum(init_dist);

Problem = GazeProblem(Parameters, SolverOptions, init_dist);

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
    [~, c1(:, i)] = sim_meas_uncertainty(Problem, init_dist);
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
    [~, c2(:, i)] = sim_meas_uncertainty(Problem,  init_dist);
end
time = toc;

fprintf('\tFinished in %fs\n', time);


%%

rng(0);

c3 = zeros(Parameters.Horizon + 1, samples);
Problem.Controller.Estimator = 'Code';

fprintf('Running N = %d simulations w/ code estimator...\n', samples);

tic;
parfor i = 1:samples
    [~, c3(:, i)] = sim_meas_uncertainty(Problem,  init_dist);
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

h1 = plot(1:size(c1, 1), mean(c1, 2), '-k');
h2 = plot(1:size(c2, 1), mean(c2, 2), '-b');
h3 = plot(1:size(c3, 1), mean(c3, 2), '-r');

shadedErrorBar(1:size(c1, 1), mean(c1, 2), std(c1, 0, 2), 'lineprops', '-k');
shadedErrorBar(1:size(c2, 1), mean(c2, 2), std(c2, 0, 2), 'lineprops', '-b');
shadedErrorBar(1:size(c3, 1), mean(c3, 2), std(c3, 0, 2), 'lineprops', '-r');

plot(1:size(c1, 1), mean(c1, 2), '-k', 'LineWidth', 2);
plot(1:size(c2, 1), mean(c2, 2), '-b', 'LineWidth', 2);
plot(1:size(c3, 1), mean(c3, 2), '-r', 'LineWidth', 2);

scatter(1:size(c1, 1), mean(c1, 2), 'k', 'filled');
scatter(1:size(c2, 1), mean(c2, 2), 'b', 'filled');
scatter(1:size(c3, 1), mean(c3, 2), 'r', 'filled');

legend([h1 h2 h3], 'Exact', 'Info State', 'Info Code');

%%

figure;
hold on;
plot(obj_hist, '-k', 'LineWidth', 2);
scatter(1:length(obj_hist), obj_hist, 'k', 'filled');
xlabel('Iter');
ylabel('Objective');
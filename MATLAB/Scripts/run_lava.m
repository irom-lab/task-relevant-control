clear;
close all;
clc;

%%

rng(0);

samples = 500;

Parameters.Length = 4;
Parameters.Goal = 3;
Parameters.Horizon = 10;
Parameters.MeasurementRate = 0.5;

SolverOptions.Tradeoff = 0.01;
SolverOptions.Iters = 30;
SolverOptions.NumCodewords = 3;
SolverOptions.FixedCodeMap = true;

init_dist = [0.3; 0.4; 0; 0.3; 0];
init_dist = init_dist ./ sum(init_dist);

Problem = LavaProblem(Parameters, SolverOptions, init_dist);

beta_range = [1 0.001];
expected_factor = 0.03;

%%

disp('Solving problem exactly...');

tic;
[controller, exact_objective] = solve_exact(Problem);
time = toc;

fprintf('\tObjective: %f\n', exact_objective);
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
betas = linspace(beta_range(1), beta_range(2), 10);
best_mi = inf;
BestProblem = [];

for i = 1:length(betas)
    fprintf('Round %d of %d, beta = %f\n', i, length(betas), betas(i));
    Problem.SolverOptions.Tradeoff = betas(i);
    [~, objective, obj_hist, expected, mi] = solve_info(Problem);
    
    if mi < best_mi && expected < exact_objective * expected_factor
        best_mi = mi;
        best_beta = betas(i);
        BestProblem = copy(Problem);
    end
end
time = toc;

Problem = copy(BestProblem);

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


fig = figure;
hold on;

title(['\beta = ' num2str(Problem.SolverOptions.Tradeoff)]);
xlabel('Timestep', 'FontSize', 15);
ylabel('Avg. Cummulative Cost', 'FontSize', 15);
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

legend([h1 h2 h3], {'Full State Solution', 'Task Driven Solution', 'TDV Estimates'}, 'FontSize', 12);

set(gcf,'Units','inches');
screenposition = get(gcf,'Position');
set(gcf,...
    'PaperPosition',[0 0 screenposition(3:4)],...
    'PaperSize',[screenposition(3:4)]);
print -dpdf -painters Figures/lava_result.pdf
savefig(fig, 'Figures/lava_result.fig', 'compact');

%%

figure;
hold on;
plot(obj_hist, '-k', 'LineWidth', 2);
scatter(1:length(obj_hist), obj_hist, 'k', 'filled');
xlabel('Iter');
ylabel('Objective');
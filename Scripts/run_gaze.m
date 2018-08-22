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

SolverOptions.Tradeoff = 20;
SolverOptions.Iters = 10;
SolverOptions.NumCodewords = 8;
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

%%

ml_code = zeros(size(Problem.Transitions, 1), 1);
angle = zeros(size(Problem.Transitions, 1), 1);
entropy = zeros(size(Problem.Transitions, 1), 1);

for i = 1:size(Problem.Transitions, 1)
    [~, ml_code(i)] = max(Problem.Controller.CodeGivenState(:, i));
    [x, ballx, bally, ballvel] = ind2sub([Parameters.Runway Parameters.Runway Parameters.Height 2], i);
    angle(i) = atan2(bally, ballx - x);
    entropy(i) = -sum(Problem.Controller.CodeGivenState(:, i) .* log(Problem.Controller.CodeGivenState(:, i)));
end

[angle, inds] = sort(angle);
ml_code = ml_code(inds);

relabeling = zeros(max(ml_code));
code_count = 1;
relabeled = false(max(ml_code));

for i = 1:length(angle)
    if ~relabeled(ml_code(i))
        relabeling(ml_code(i)) = code_count;
        code_count = code_count + 1;
        relabeled(ml_code(i)) = true;
    end
end

%%

figure;
hold on;

plot(cos(linspace(0, pi)), sin(linspace(0, pi)), 'k', 'LineWidth', 2);

for i = 1:SolverOptions.NumCodewords
    r = 1 + 0.1 * i;
    scatter(r * cos(angle(ml_code == i)), r * sin(angle(ml_code == i)), 'filled');
end

mark_angles = deg2rad(0:15:180);

for i = 1:length(mark_angles)
    plot([0.99 * cos(mark_angles(i)); 1.01 * cos(mark_angles(i))], [0.99 * sin(mark_angles(i)); 1.01 * sin(mark_angles(i))], 'k', 'LineWidth', 2);
end



pbaspect([1 0.5 1]);
xlim([-2 2]);
ylim([0 2]);
title(['Codewords on Unit Sphere with \beta = ' num2str(SolverOptions.Tradeoff)]);

%%

figure;
hold on;

scatter(angle, relabeling(ml_code), 'k', 'filled');
close all;
clear;
clc;

%%

rng(0);

n = 2;
m = 2;

Init.mean = [5; 3];
Init.cov = 1 * eye(2);

Parameters.A = rand(n, n);
Parameters.B = rand(n, m);
Parameters.Q = diag([100 1]);
Parameters.R = eye(m);
Parameters.MeasCov = 1 * diag([1 1]);
Parameters.ProcCov = 1 * diag([0.1, 0.1]);

Parameters.NStates = 2;
Parameters.NInputs = 2;

Parameters.Horizon = 5;
Parameters.Goals = zeros(2, Parameters.Horizon);


SolverOptions.Iters = 7;
SolverOptions.Tradeoff = 0.0001;
SolverOptions.NumCodewords = 2;
SolverOptions.FixedCodeMap = false;
SolverOptions.Tol = 1e-6;

samples = 1000;

Problem = LTIProblem(Parameters, SolverOptions, Init);

%%

[controller, obj_val, obj_hist] = solve_exact(Problem);

%%
fprintf('Running N = %d simulations...\n', samples);

rng(0);

c1 = zeros(Parameters.Horizon + 1, samples);

tic;
for i = 1:samples
    [~, c1(:, i)] = sim_meas_uncertainty(Problem, Parameters.Horizon);
end

time = toc;



%%

fprintf('Running N = %d simulations...\n', samples);

rng(0);

c2 = zeros(Parameters.Horizon + 1, samples);

tic;
for i = 1:samples
    [~, c2(:, i)] = sim_meas_uncertainty(Problem, Parameters.Horizon, Parameters.MeasCov);
end

time = toc;


%%

[controller, obj_val, obj_hist, mean_traj] = solve_info(Problem);

%%

fprintf('Running N = %d simulations...\n', samples);

rng(0);

c3 = zeros(Parameters.Horizon + 1, samples);
traj = zeros(2, Parameters.Horizon + 1, samples);

tic;
for i = 1:samples
    [traj(:, :, i), c3(:, i)] = sim_meas_uncertainty(Problem, Parameters.Horizon);
end

time = toc;

%%

figure;
hold on;

title(['\beta = ' num2str(SolverOptions.Tradeoff)]);
xlabel('Timestep', 'FontSize', 15);
ylabel('Cummulative Cost', 'FontSize', 15);
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

legend([h1 h2 h3], {'Incorrect Covariance LQG', 'Correct Covariance LQG', 'Incorrect Covariance TRV'});

set(gcf,'Units','inches');
screenposition = get(gcf,'Position');
set(gcf,...
    'PaperPosition',[0 0 screenposition(3:4)],...
    'PaperSize',[screenposition(3:4)]);
print -dpdf -painters fig
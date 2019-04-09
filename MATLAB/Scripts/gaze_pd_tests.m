close all
clear
clc

%%
Parameters.g = 9.8;
Parameters.center_y = 0;
Parameters.f = 10;

delta_t = 1 / 60;

ic = [0; 5.5; 4; 0; -1.5; 0];
y0 = gaze_outputs(0, ic, Parameters);

%%
figure;

%% Gains Heuristic

kp = 10; %10
kd = 5;  %5

kc1 = 0;

kc2 = 10;
kd2 = 10;

%% Controllers

u = {@(y) kp * (y0(1) - y(1, end)) - kd * y(2, end)
     @(y) -kc1 * (y(2, end) - y(2, end - 1)) / delta_t};
 
titles = {'Gaze', 'Chapman'};
 
%% Simulate

x = [ic' gaze_outputs(0, ic', Parameters)'];
t = 0;
first = true;

velocities = {};
times = {};

for controller_idx = 1:length(u)
    x = [ic' gaze_outputs(0, ic', Parameters)'];
    t = 0;
    first = true;
    
    while true
        if first
            x(end + 1, 1:6) = x(end, 1:6) + ...
                delta_t * gaze_dynamics(t(end), x', @(y) 0, Parameters)';
            first = false;
        else
            x(end + 1, 1:6) = x(end, 1:6) + ...
                delta_t * gaze_dynamics(t(end), x', u{controller_idx}, Parameters)';
        end
        
        x(end, 7:end) = gaze_outputs(t, x(end - 1, :), Parameters)';

        t(end + 1) = t(end) + delta_t;

        if x(end, 3) <= 0
            break
        end
    end
    
    velocities{end + 1} = x(:, 4);
    times{end + 1} = t;
    
    subplot(1, length(u), controller_idx)
    title(titles{controller_idx});
    plot_gaze(t, x);
end

figure;
hold on;
title('Velocities');

for i = 1:length(velocities)
    plot(times{i}, velocities{i}, 'LineWidth', 2)
end

%%

function dxdt = gaze_dynamics(t, x, u, Parameters)

input = u(x(7:8, :));

%if input > 4
%    input = 4;
%end

x = x(:, end);

y = gaze_outputs(t, x, Parameters);

dxdt = [x(4:6); input; 0; -Parameters.g];
%dxdt = [x(4:6); input; 0; -Parameters.g];
end

function y = gaze_outputs(t, x, Parameters)

d = x(1);
bx = x(2);
by = x(3);
d_dot = x(4);
bx_dot = x(5);
by_dot = x(6);

y = [Parameters.f * by / (bx - d) + Parameters.center_y;
    Parameters.f * (by_dot * (bx - d) + by * (d_dot - bx_dot)) / ((bx - d) ^ 2)];
end

function [value, terminal, dir] = gaze_events(t, x)
value = x(3);
terminal = true;
dir = -1;
end

function plot_gaze(t, x)
hold on;

plot(x(:, 2), x(:, 3), 'k-', 'LineWidth', 2)
plot(x(:, 1), zeros(size(x, 1), 1), 'b-', 'LineWidth', 2)

scatter(x(end, 2), 0, 'ko', 'filled', 'LineWidth', 2)
scatter(x(end, 1), 0, 'bo', 'filled', 'LineWidth', 2)
end
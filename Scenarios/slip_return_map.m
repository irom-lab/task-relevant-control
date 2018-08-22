function next_stance = slip_return_map(Stance, Parameters)
%SIM_HOP Simulates one hop of the SLIP model.
%   Init is a state vector in stance coordinates [r; theta; rdot; thetadot]

options = odeset('Events', @(t, x) stance_events(t, x, Parameters));

[t, traj] = ode45(@(t, x) stance_dynamics(t, x, Parameters), ...
    [0, Parameters.TMax], Stance, options);

final_stance_state = traj(end, :)';
init_flight_state = stance_to_flight(final_stance_state, Parameters);

options = odeset('Events', @(t, x) flight_events(t, x, Parameters));

[t, traj] = ode45(@(t, x) flight_dynamics(t, x, Parameters), ...
    [t(end), Parameters.TMax], init_flight_state, options);

final_flight_state = traj(end, :)';
next_stance = flight_to_stance(final_flight_state, Parameters);

end

function [value, isterminal, direction] = stance_events(t, x, Parameters)
    r0 = Parameters.TouchdownRadius;

    value = x(1) - r0;
    isterminal = true;
    direction = 1;
end

function [value, isterminal, direction] = flight_events(t, x, Parameters)
    r0 = Parameters.TouchdownRadius;
    theta0 = Parameters.TouchdownAngle;

    value = x(2) - r0 * cos(theta0);
    isterminal = true;
    direction = -1;
end

% === Flight-to-stance dynamics ===================
function xp = flight_to_stance(x, Parameters)
% x = [x;y;xdot;ydot]
% xp = [r,theta,rdot,thetadot]

m = Parameters.Mass;
r0 = Parameters.TouchdownRadius;
theta0 = Parameters.TouchdownAngle;
k = Parameters.SpringConst;
g = Parameters.Gravity;

r = x(2)/cos(theta0);

xp = [ r ; theta0; ...
  -x(3)*sin(theta0) + x(4)*cos(theta0); ...
  -(x(3)*cos(theta0) + x(4)*sin(theta0))/r];
end

% === Stance-to-flight dynamics ================
function x = stance_to_flight(xp, Parameters)
% x = [x;y;xdot;ydot]
% xp = [r,theta,rdot,thetadot]

m = Parameters.Mass;
r0 = Parameters.TouchdownRadius;
theta0 = Parameters.TouchdownAngle;
k = Parameters.SpringConst;
g = Parameters.Gravity;

x = [ -xp(1)*sin(xp(2));...
  xp(1)*cos(xp(2)); ...
  -xp(3)*sin(xp(2)) - xp(1)*xp(4)*cos(xp(2)); ...
  xp(3)*cos(xp(2)) - xp(1)*xp(4)*sin(xp(2))];

end
  
% === Flight dynamics ================================
function xdot = flight_dynamics(t, x, Parameters)
% x = [x;y;xdot;ydot]

m = Parameters.Mass;
r0 = Parameters.TouchdownRadius;
theta0 = Parameters.TouchdownAngle;
k = Parameters.SpringConst;
g = Parameters.Gravity;

xdot = [x(3:4); 0; -g/m];

end

% === Stance dynamics ===============================
function xpdot = stance_dynamics(t, xp, Parameters)
% xp = [r,theta,rdot,thetadot]

m = Parameters.Mass;
r0 = Parameters.TouchdownRadius;
theta0 = Parameters.TouchdownAngle;
k = Parameters.SpringConst;
g = Parameters.Gravity;

xpdot = [xp(3:4); ...
  k/m*(r0-xp(1)) + xp(1)*xp(4)^2 - g*cos(xp(2)); ... 
  (g*xp(1)*sin(xp(2)) - 2*xp(1)*xp(3)*xp(4))/(xp(1)^2)];
  
end
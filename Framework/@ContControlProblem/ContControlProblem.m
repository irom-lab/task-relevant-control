classdef (Abstract) ContControlProblem < ControlProblem
    %CONTCONTROLPROBLEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Init
        Controller
        NStates
        NInputs
    end
    
    methods
        function obj = ContControlProblem(Parameters, SolverOptions, Init, Controller)
            obj = obj@ControlProblem(Parameters, SolverOptions);
            
            if nargin > 2
                obj.Init = Init;
            else
                obj.Init = [];
            end

            if nargin > 3
                obj.Controller = Controller;
            else
                obj.Controller = {};
            end
        end
        
        function [traj, costs] = simulate(Obj, horizon)
            if isfinite(Obj.Parameters.Horizon) && nargin < 2
                horizon = Obj.Parameters.Horizon;
            end
            
            traj = [Obj.Init zeros(Obj.NStates, horizon)];
            costs = zeros(horizon + 1, 1);
            
            for t = 1:horizon
                costs(t) = cost(Obj, traj(:, t), Obj.Controller(:, :, t) * traj(:, t));
                traj(:, t + 1) = dynamics(Obj, traj(:, t), Obj.Controller(:, :, t) * traj(:, t));
            end
            
            costs(end) = terminal_cost(Obj, traj(:, end));
        end
    end
    
    
    methods (Abstract)
        next_state = dynamics(Obj, State, Input);
        [A, B] = linearize(Obj, State, Input)
        
        c = cost(Obj, State, Input)
        [Q, R] = quadraticize_cost(Obj, State, Input)
        
        c = terminal_cost(State);
        Q = quadraticize_terminal_cost(Obj, State)
    end
end


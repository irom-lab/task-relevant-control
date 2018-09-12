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
            
            traj = [Obj.Init.mean zeros(Obj.NStates, horizon)];
            costs = zeros(horizon + 1, 1);
            
            for t = 1:horizon
                input = Obj.Controller.K(:, :, t) * traj(:, t) + Obj.Controller.d(:, t);
                costs(t) = cost(Obj, traj(:, t), input);
                traj(:, t + 1) = dynamics(Obj, traj(:, t), input);
            end
            
            costs(end) = terminal_cost(Obj, traj(:, end));
        end
        
        [ traj, cum_cost ] = sim_meas_uncertainty(Obj, init, horizon)
        
        [ controller, obj_val, obj_hist ] = solve_info_ilqr(Obj)
        
        [ controller, obj_val, obj_hist ] = solve_info_lqg(Obj)
    end
    
    
    methods (Abstract)
        next_state = dynamics(Obj, State, Input, t)
        [A, B] = linearize(Obj, State, Input, t)
        
        c = cost(Obj, State, Input, t)
        [Q, R] = quadraticize_cost(Obj, State, Input, t)
        
        c = terminal_cost(State);
        Q = quadraticize_terminal_cost(Obj, State)
    end
end


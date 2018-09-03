classdef SLIPProblem < ContControlProblem
    %LINCONTROLPROBLEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = SLIPProblem(Parameters, SolverOptions, Init, varargin)
            obj = obj@ContControlProblem(Parameters, SolverOptions, Init, varargin{:});
            
            obj.Parameters.NStates = 5;
            obj.Parameters.NInputs = 1;
        end
    end
    
    methods
        function next_state = dynamics(Obj, State, Input, t)
            next_state = slip_return_map(State, Input, Obj.Parameters);
        end
        
        [A, B] = linearize(Obj, State, Input, t)
        
        function c = cost(Obj, State, Input, t)
            c = State' * Obj.Parameters.Q(:, :, t) * State + Input' * Obj.Parameters.R(:, :, t) * Input;
        end
        
        function [Q, R] = quadraticize_cost(Obj, State, Input, t)
            Q = Obj.Parameters.Q(:, :, t);
            R = Obj.Parameters.R(:, :, t);
        end
        
        function c = terminal_cost(Obj, State)
            c = State' * Obj.Parameters.Q(:, :, end) * State;
        end
        
        function Q = quadraticize_terminal_cost(Obj, State)
            Q = Obj.Parameters.Q(:, :, end);
        end
    end
end


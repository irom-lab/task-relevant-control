classdef LTIProblem < ContControlProblem
    %LINCONTROLPROBLEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = LTIProblem(Parameters, SolverOptions, Init, varargin)
            obj = obj@ContControlProblem(Parameters, SolverOptions, Init, varargin{:});
            
            obj.NStates = size(Parameters.A, 1);
            obj.NInputs = size(Parameters.B, 2);
        end
    end
    
    methods
        function next_state = dynamics(Obj, State, Input, t)
            next_state = Obj.Parameters.A * State + Obj.Parameters.B * Input;
        end
        
        function [A, B] = linearize(Obj, State, Input, t)
            A = Obj.Parameters.A;
            B = Obj.Parameters.B;
        end
        
        function c = cost(Obj, State, Input, t)
            c = (State - Obj.Parameters.Goals(:, t))' * Obj.Parameters.Q * (State - Obj.Parameters.Goals(:, t)) ...
                + Input' * Obj.Parameters.R * Input;
        end
        
        function [Q, R] = quadraticize_cost(Obj, State, Input, t)
            Q = Obj.Parameters.Q;
            R = Obj.Parameters.R;
        end
        
        function c = terminal_cost(Obj, State)
            c = (State - Obj.Parameters.Goals(:, end))' * Obj.Parameters.Q * (State - Obj.Parameters.Goals(:, end));
        end
        
        function Q = quadraticize_terminal_cost(Obj, State)
            Q = Obj.Parameters.Q;
        end
    end
end


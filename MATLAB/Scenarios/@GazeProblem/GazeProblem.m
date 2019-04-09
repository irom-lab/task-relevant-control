classdef GazeProblem < ContControlProblem
    %GAZEPROBLEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = GazeProblem(Parameters, SolverOptions, Init, varargin)
            obj = obj@ContControlProblem(Parameters, SolverOptions, Init, varargin{:});
            
            obj.Parameters.NStates = 7;
            obj.Parameters.NInputs = 1;
        end
    end
    
    methods
        function next_state = dynamics(Obj, State, Input, t)
            next_state = State + Obj.Parameters.DeltaT * [State(4:6); Input; 0; -Obj.Parameters.Gravity; 0];
        end
        
        function [A, B] = linearize(Obj, State, Input, t)
            dt = Obj.Parameters.DeltaT;
            g = Obj.Parameters.Gravity;
            
            A = [1 0 0 dt 0 0 0;
                 0 1 0 0 dt 0 0;
                 0 0 1 0 0 dt 0;
                 0 0 0 1 0 0 0;
                 0 0 0 0 1 0 0;
                 0 0 0 0 0 1 -dt * g;
                 0 0 0 0 0 0 1];
             
            B = [zeros(3, 1); 1; zeros(3, 1)];
        end
        
        function c = cost(Obj, State, Input, t)
            c = Input' * Obj.Parameters.R(:, :, t) * Input;
        end
        
        function [Q, R] = quadraticize_cost(Obj, State, Input, t)
            Q = Obj.Parameters.Q(:, :, t);
            R = Obj.Parameters.R(:, :, t);
        end
        
        function c = terminal_cost(Obj, State)
            c = (State - Obj.Parameters.Goals(:, end))' * Obj.Parameters.Q(:, :, end) * (State - Obj.Parameters.Goals(:, end));
        end
        
        function Q = quadraticize_terminal_cost(Obj, State)
            Q = Obj.Parameters.Q(:, :, end);
        end
    end
    
end


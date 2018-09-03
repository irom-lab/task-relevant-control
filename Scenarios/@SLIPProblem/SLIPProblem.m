classdef SLIPProblem < ContControlProblem
    %LINCONTROLPROBLEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = SLIPProblem(Parameters, SolverOptions, Init, varargin)
            obj = obj@ContControlProblem(Parameters, SolverOptions, Init, varargin{:});
            
            obj.NStates = size(Parameters.A, 3);
            obj.NInputs = size(Parameters.B, 1);
        end
    end
    
    methods
        function next_state = dynamics(Obj, State, Input)
            stance = [State(1); Input State(2:3)];
            next_stance = slip_return_map(stance, Obj.Parameters);
            next_state = [next_stance(1) next_stance(3:4)];
        end
        
        [A, B] = linearize(Obj, State, Input)
        
        function c = cost(Obj, State, Input)
            c = State' * Obj.Parameters.Q * State + Input' * Obj.Parameters.R * Input;
        end
        
        function [Q, R] = quadraticize_cost(Obj, State, Input)
            Q = Obj.Parameters.Q;
            R = Obj.Parameters.R;
        end
        
        function c = terminal_cost(Obj, State)
            c = State' * Obj.Parameters.Q * State;
        end
        
        function Q = quadraticize_terminal_cost(Obj, State)
            Q = Obj.Parameters.Q;
        end
    end
end


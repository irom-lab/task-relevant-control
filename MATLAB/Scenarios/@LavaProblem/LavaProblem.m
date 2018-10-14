classdef LavaProblem < DiscControlProblem
    %LAVAPROBLEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = LavaProblem(Parameters, SolverOptions, varargin)
            obj = obj@DiscControlProblem(Parameters, SolverOptions, varargin{:});
        end
        
        
        function [costs, terminal_costs] = costs(Obj)
            costs = zeros(Obj.Parameters.Length + 1, 3);
            
            costs(1:Obj.Parameters.Goal, 1) = 1;
            costs(Obj.Parameters.Goal + 1, 1) = -5;
            costs((Obj.Parameters.Goal + 2):end, 1) = 1;
            
            costs(Obj.Parameters.Goal, 2) = -5;
            
            costs(1:(Obj.Parameters.Goal - 2), 3) = 1;
            costs(Obj.Parameters.Goal - 1, 3) = -5;
            costs(Obj.Parameters.Goal:end, 3) = 1;
            
            terminal_costs = zeros(Obj.Parameters.Length + 1, 1);
            terminal_costs(Obj.Parameters.Goal) = -10;
            terminal_costs(end) = 10;
        end
        
        function trans = transitions(Obj)
            trans = zeros(Obj.Parameters.Length + 1, Obj.Parameters.Length + 1, 3);
            
            trans(1, 1, 1) = 1;
            trans(1, 1, 2) = 1;
            trans(2, 1, 3) = 1;
            
            trans(end, end, :) = 1;
            
            for i = 2:Obj.Parameters.Length
                trans(i - 1, i, 1) = 1;
                trans(i, i, 2) = 1;
                trans(i + 1, i, 3) = 1;
            end
        end
        
        function output_given_state = sensor(Obj)
            n = size(Obj.Transitions, 1);
            error_prob = (1 - Obj.Parameters.MeasurementRate) / (n - 1);
            output_given_state = Obj.Parameters.MeasurementRate .* eye(n) + error_prob .* (ones(n) - eye(n));
        end
    end
    
end


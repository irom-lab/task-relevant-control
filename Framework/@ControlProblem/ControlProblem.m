classdef ControlProblem < handle
    %CONTROLPROBLEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Parameters
        SolverName
        SolverOptions
    end
    
    methods
        function obj = ControlProblem(Parameters, SolverOptions)
            obj.Parameters = Parameters;
            obj.SolverName = '';
            
            if nargin > 1
                obj.SolverOptions = SolverOptions;
            else
                obj.SolverOptions = {};
            end
        end
    end
    
end


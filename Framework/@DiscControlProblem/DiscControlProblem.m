classdef (Abstract) DiscControlProblem < ControlProblem
    %DISCCONTROLPROBLEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Controller
        Init
        Costs %(state, input)
        TerminalCosts
        Transitions %(next state, state, input)
        Sensor
    end
    
    properties (SetAccess = protected)
        ControllerType
    end
    
    methods
        function obj = DiscControlProblem(Parameters, SolverOptions, Init, Controller)
            obj = obj@ControlProblem(Parameters, SolverOptions);
            
            [obj.Costs, obj.TerminalCosts] = costs(obj);
            obj.Transitions = transitions(obj);
            
            obj.Sensor = sensor(obj);
            
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
    end
    
    methods (Abstract)
        costs
        transitions
        sensor
    end
end


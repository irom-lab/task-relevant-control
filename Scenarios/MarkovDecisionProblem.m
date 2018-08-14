classdef MarkovDecisionProblem
    %MDPDESCRIPTION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        ScenarioType
        ScenarioID
        Scenario
        Horizon
        ObjectiveType
        Solver
        SolverOptionsID = 0;
        SolverOptions = {};
        Transitions;
        Costs;
        TerminalCosts;
    end
    
    methods
        function obj = MarkovDecisionProblem(ScenarioType, ScenarioID, Horizon, ObjectiveType, Solver, SolverOptionsID)
            obj.ScenarioType = ScenarioType;
            obj.ScenarioID = ScenarioID;
            obj.Horizon = Horizon;
            obj.ObjectiveType = ObjectiveType;
            obj.Solver = Solver;
            
            rmpath(genpath('Scenarios'));
            addpath(['Scenarios/' ScenarioType]);
            
            if nargin == 6
                obj.SolverOptionsID = SolverOptionsID;
                obj.SolverOptions = load_options(SolverOptionsID);
            end
            
            obj.Scenario = load_scenario(obj.ScenarioID);
            
            [P, R, obj.TerminalCosts] = scenario2mdp(obj.Scenario);
            obj.Transitions = permute(P, [2, 3, 1]);
            obj.Costs = permute(R, [2, 3, 1]);
        end
    end
end


classdef (Abstract) ContControlProblem < ControlProblem
    %CONTCONTROLPROBLEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Init
        Controller
    end
    
    methods
        function obj = ContControlProblem(Parameters, SolverName, SolverOptions, Init, Controller)
            obj = obj@ControlProblem(Parameters, SolverName, SolverOptions);
            
            if nargin > 3
                obj.Init = Init;
            else
                obj.Init = [];
            end

            if nargin > 4
                obj.Controller = Controller;
            else
                obj.Controller = {};
            end
        end
    end
    
    methods
        function [traj, times] = simulate(Obj, Time, Init)
            if nargin < 3
                init = Obj.Init;
            else
                init = Init;
            end
            
            options = odeset('Events', @(t, x) transconds(Obj, t, x));
            
            autonomous = @(t, x) dynamics(Obj, t, x, Obj.Controller(t, x), options);
            
            t0 = 0;
            traj = [];
            times = [];
            
            while true
                [segtimes, segtraj, eventtime, eventstate, eventidx] = ode45(autonomous, [t0, Time], init);
                
                if ~isempty(eventtime) && eventtime(1) < Time
                    init = transfunction(Obj, eventstate(1, :), eventidx);
                    times = [times; segtimes(segtimes < eventtime(1))];
                    traj = [traj; segtraj(segtimes < eventtime(1))];
                end
                
                if times(end) > Time
                    traj = traj(times < Time);
                    times = times(times < Time);
                end
            end
        end
    end
    
    methods (Abstract)
        dynamics
        linearize
        transconds
        transfunction
    end
end


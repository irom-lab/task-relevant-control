classdef GazeProblem < DiscControlProblem
    %GAZEPROBLEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = GazeProblem(Parameters, SolverOptions, varargin)
            obj = obj@DiscControlProblem(Parameters, SolverOptions, varargin{:});
        end
        
        
        function [costs, terminal_costs] = costs(Obj)
            runway = Obj.Parameters.Runway;
            height = Obj.Parameters.Height;
            
            n = runway * runway * height * 2;
            m = 5;
            
            costs = [ones(n, 4) zeros(n, 1)];
            terminal_costs = zeros(n, 1);
            
            for x = 1:runway
                for ballx = 1:runway
                    for ballvel = 1:2
                        if ballx ~= x
                            state_ind = sub2ind([runway runway height 2], x, ballx, 1, ballvel);
                            terminal_costs(state_ind) = 0;
                        else
                            state_ind = sub2ind([runway runway height 2], x, ballx, 1, ballvel);
                            terminal_costs(state_ind) = -20;
                        end
                    end
                end
            end
        end
        
        function trans = transitions(Obj)
            runway = Obj.Parameters.Runway;
            height = Obj.Parameters.Height;
            
            n = runway * runway * height * 2;
            m = 5;
            
            trans = zeros(n, n, m);
            
            for x = 1:runway
                for ballx = 1:runway
                    for bally = 1:height
                        for ballvel = 1:2
                            for input = 1:m
                                if height > 1
                                    switch input
                                        case 1
                                            next_state = {x - 1, ballx - ballvel, bally - 1, ballvel};
                                        case 2
                                            next_state = {x - 2, ballx - ballvel, bally - 1, ballvel};
                                        case 3
                                            next_state = {x + 1, ballx - ballvel, bally - 1, ballvel};
                                        case 4
                                            next_state = {x + 2, ballx - ballvel, bally - 1, ballvel};
                                        case 5
                                            next_state = {x, ballx - ballvel, bally - 1, ballvel};
                                    end
                                else
                                    next_state = {x, ballx, bally, ballvel};
                                end
                                
                                if next_state{1} < 1
                                    next_state{1} = 1;
                                end
                                
                                if next_state{1} > runway
                                    next_state{1} = runway;
                                end
                                
                                if next_state{2} < 1
                                    next_state{2} = runway;
                                end
                                
                                if next_state{3} < 1
                                    next_state{3} = 1;
                                end
                                
                                state_ind = sub2ind([runway runway height 2], x, ballx, bally, ballvel);
                                next_state_ind = sub2ind([runway runway height 2], next_state{:});
                                
                                trans(next_state_ind, state_ind, input) = 1;
                            end
                        end
                    end
                end
            end
        end
        
        function output_given_state = sensor(Obj)
            n = size(Obj.Transitions, 1);
            runway = Obj.Parameters.Runway;
            height = Obj.Parameters.Height;
            
            output_given_state = Obj.Parameters.MeasurementRate * eye(n);
            
            for i = 1:n
                [x, ballx, bally, ballvel] = ind2sub([runway runway height 2], i);
                inds = [];
                
                if ballvel == 1
                    inds = sub2ind([runway runway height 2], x, ballx, bally, 2);
                else
                    inds = sub2ind([runway runway height 2], x, ballx, bally, 1);
                end
                
                if x < runway
                    inds = [inds; sub2ind([runway runway height 2], x + 1, ballx, bally, ballvel)];
                end
                
                if x > 1
                    inds = [inds; sub2ind([runway runway height 2], x - 1, ballx, bally, ballvel)];
                end
                
                if ballx < runway
                    inds = [inds; sub2ind([runway runway height 2], x, ballx + 1, bally, ballvel)];
                end
                
                if ballx > 1
                    inds = [inds; sub2ind([runway runway height 2], x, ballx - 1, bally, ballvel)];
                end
                
                if bally < height
                    inds = [inds; sub2ind([runway runway height 2], x, ballx, bally + 1, ballvel)];
                end
                
                if bally > 1
                    inds = [inds; sub2ind([runway runway height 2], x, ballx, bally - 1, ballvel)];
                end
                
                output_given_state(inds, i) = (1 - Obj.Parameters.MeasurementRate) / length(inds);
            end
        end
    end
    
end


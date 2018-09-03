function [A, B] = linearize(Obj, State, Input)
    delta = Obj.Parameters.Delta;
    
    A = zeros(Obj.Parameters.NStates);
    B = 0;
    
    for i = 1:Obj.Parameters.NStates
        x = State;
        x(i) = x(i) + delta;
        forward = dynamics(Obj, x, Input);
        
        x = State;
        x(i) = x(i) - delta;
        reverse = dynamics(Obj, x, Input);
        
        A(:, i) = (forward - reverse) / (2 * delta);
    end
    
    forward = dynamics(Obj, State, Input + delta);
    reverse = dynamics(Obj, State, Input - delta);
    B = (forward - reverse) / (2 * delta);
end
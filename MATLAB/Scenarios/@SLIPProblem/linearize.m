function [A, B] = linearize(Obj, State, Input, t)
    delta = 1e-2;%Obj.Parameters.Delta; I changed this for debugging the python version.
    
    A = zeros(Obj.Parameters.NStates);
    B = 0;
    
    for i = 1:Obj.Parameters.NStates
        x = State;
        x(i) = x(i) + delta;
        forward = dynamics(Obj, x, Input, t);
        
        x = State;
        x(i) = x(i) - delta;
        reverse = dynamics(Obj, x, Input, t);
        
        A(:, i) = (forward - reverse) / (2 * delta);
    end
    
    forward = dynamics(Obj, State, Input + delta, t);
    reverse = dynamics(Obj, State, Input - delta, t);
    B = (forward - reverse) / (2 * delta);
end
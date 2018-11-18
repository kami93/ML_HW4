classdef log_sigmoid_cross_entropy < handle
    % SHPark, SPA lab, Hanyang University. 2018.
    properties
        logit
        error
        delta
    end
    
    methods
        function obj = log_sigmoid_cross_entropy(x, y)
            obj.logit = sigmoid(x);
            obj.error = -y .* log(obj.logit) + (y - 1) .* log(1 - obj.logit);
            obj.delta = obj.logit - y;
        end
    end
end


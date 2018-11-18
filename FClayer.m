classdef FClayer < handle
    % SHPark, SPA lab, Hanyang University. 2018.
    properties
        hidden
        delta
        kernel
        bias
        units
        activation
        use_bias
        kernel_initializer
        bias_initializer
        built
    end
    
    methods
        function obj = FClayer(units, activation, use_bias, kernel_initializer, bias_initializer)
            % units
            if ~exist("units", 'var')
                error('Output units must be specified')
            elseif isempty(units)
                error('Output units must be specified')
            end
            if isnumeric(units)
                obj.units = units;
            else
                error('Output units must be numeric')
            end
            % activation
            if ~exist("activation", 'var')
                obj.activation = @relu;
            elseif isempty(activation)
                obj.activation = @relu;
            end
            if isstring(activation)
                if ismember(activation, ["tanh", "relu", "sigmoid", "linear"])
                    if activation == "linear"
                       obj.activation = activation;
                    else
                    obj.activation = str2func(activation);
                    end
                else
                    error('Activation must be either "tanh", "relu", "sigmoid", or "linear".')
                end
            else
                error('Activation must be specified as string.')
            end
           
            if ~exist("use_bias", 'var')
                obj.use_bias = true;
            elseif isempty(use_bias)
                obj.use_bias = true;
            else
                obj.use_bias = use_bias;
            end
            % kernel initializer
            if ~exist("kernel_initializer", 'var')
                obj.kernel_initializer = @randn;
            elseif isempty(kernel_initializer)
                obj.kernel_initializer = @randn;
            elseif isstring(kernel_initializer)
                if kernel_initializer == "random_normal"
                    obj.kernel_initializer = @randn;
                elseif kernel_initializer == "random_uniform"
                    obj.kernel_initializer = @rand;
                elseif kernel_initializer == "ones"
                    obj.kernel_initializer = @ones;
                else
                    error('kernel initializer must be either "random_normal", "random_uniform", or "ones".')
                end
            else
                error('kernel initializer must be specified as string.')
            end
            % bias initializer
            if ~exist("bias_initializer", 'var')
                obj.bias_initializer = @randn;
            elseif isempty(bias_initializer)
                obj.bias_initializer = @randn;
            elseif isstring(bias_initializer)
                if bias_initializer == "random_normal"
                    obj.bias_initializer = @randn;
                elseif bias_initializer == "random_uniform"
                    obj.bias_initializer = @rand;
                elseif bias_initializer == "ones"
                    obj.bias_initializer = @ones;
                else
                    error('bias_initializer must be either "random_normal", "random_uniform", or "ones".')
                end
            else
                error('bias_initializer must be specified as string.')
            end
            obj.built = false;
        end
   
        function built = build(obj, x)
            input_shape = size(x, 2);
            obj.kernel = obj.kernel_initializer(input_shape, obj.units);
            if obj.use_bias
                obj.bias = obj.bias_initializer(1, obj.units);
            end
            obj.built = true;
            built = true;
        end
        
        function h = forward(obj, x)
            if ~obj.built
               obj.build(x);
            end
            
            try
                obj.hidden = obj.activation(x * obj.kernel + obj.bias);
            catch
                obj.hidden = x * obj.kernel + obj.bias;
            end
            h = obj.hidden;
        end
                     
        function d = backward(obj, e, weights)
            if isequal(obj.activation, @tanh)
                obj.delta = (1 - obj.hidden.^2) .* (e * weights');
            elseif isequal(obj.activation, @relu)
                pos_mask = cast(obj.hidden > 0, 'double');
                obj.delta = pos_mask .* (e * weights');
            elseif isequal(obj.activation, @sigmoid)
                obj.delta =  (obj.hidden - obj.hidden.^2) .* (e * weights');
            else
                obj.delta = e * weights';
            end
            d = obj.delta;
        end
        
        function [w_new, b_new] = update(obj, step_size, input)
           num_batch = size(input, 1);
           obj.kernel = obj.kernel -  step_size * input' * obj.delta / num_batch;
           obj.bias = obj.bias - step_size * sum(obj.delta, 1) / num_batch;
           w_new = obj.kernel;
           b_new = obj.bias;
        end
    end
end


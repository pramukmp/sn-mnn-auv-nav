classdef MemoryNeuralNetwork
    properties
        number_of_input_neurons
        number_of_hidden_neurons
        number_of_output_neurons
        neeta
        neeta_dash
        lipschitz_norm
        spectral_norm
        seed_value
        squared_error
        rmse
        alpha_input_layer
        alpha_hidden_layer
        alpha_last_layer
        beta
        weights_input_to_hidden_nn
        weights_hidden_to_output_nn
        weights_input_to_hidden_mn
        weights_hidden_to_output_mn
        prev_output_of_input_layer_nn
        prev_output_of_input_layer_mn
        prev_output_of_hidden_layer_nn
        prev_output_of_hidden_layer_mn
        prev_output_of_nn
        prev_output_of_mn
    end

    methods
        function obj = MemoryNeuralNetwork(number_of_input_neurons, number_of_hidden_neurons, number_of_output_neurons, neeta, neeta_dash, lipschitz_norm, spectral_norm, seed_value)
            if nargin < 8
                seed_value = 16981;
            end
            
            rng(seed_value);
            
            obj.number_of_input_neurons = number_of_input_neurons;
            obj.number_of_hidden_neurons = number_of_hidden_neurons;
            obj.number_of_output_neurons = number_of_output_neurons;
            obj.squared_error = 0.0;
            obj.spectral_norm = spectral_norm;
            obj.lipschitz_norm = lipschitz_norm;
            obj.neeta = neeta;
            obj.neeta_dash = neeta_dash;
            
            obj.alpha_input_layer = rand(1, number_of_input_neurons);
            obj.alpha_hidden_layer = rand(1, number_of_hidden_neurons);
            obj.alpha_last_layer = rand(1, number_of_output_neurons);
            
            obj.beta = rand(1, number_of_output_neurons);
            obj.weights_input_to_hidden_nn = rand(number_of_input_neurons, number_of_hidden_neurons);
            obj.weights_hidden_to_output_nn = rand(number_of_hidden_neurons, number_of_output_neurons);
            obj.weights_input_to_hidden_mn = rand(number_of_input_neurons, number_of_hidden_neurons);
            obj.weights_hidden_to_output_mn = rand(number_of_hidden_neurons, number_of_output_neurons);
            
            obj.prev_output_of_input_layer_nn = zeros(1, number_of_input_neurons);
            obj.prev_output_of_input_layer_mn = zeros(1, number_of_input_neurons);
            obj.prev_output_of_hidden_layer_nn = zeros(1, number_of_hidden_neurons);
            obj.prev_output_of_hidden_layer_mn = zeros(1, number_of_hidden_neurons);
            obj.prev_output_of_nn = zeros(1, number_of_output_neurons);
            obj.prev_output_of_mn = zeros(1, number_of_output_neurons);
        end

        function output = feedforward(obj, input_array)
            obj.input_nn = input_array;
            obj.output_of_input_layer_nn = obj.input_nn;
            obj.output_of_input_layer_mn = (obj.alpha_input_layer .* obj.prev_output_of_input_layer_nn) + ((1.0 - obj.alpha_input_layer) .* obj.prev_output_of_input_layer_mn);
            
            obj.input_to_hidden_layer_nn = (obj.weights_input_to_hidden_nn' * obj.output_of_input_layer_nn') + (obj.weights_input_to_hidden_mn' * obj.output_of_input_layer_mn');
            obj.output_of_hidden_layer_nn = obj.activation_function(obj.input_to_hidden_layer_nn);
            obj.output_of_hidden_layer_mn = (obj.alpha_hidden_layer .* obj.prev_output_of_hidden_layer_nn) + ((1.0 - obj.alpha_hidden_layer) .* obj.prev_output_of_hidden_layer_mn);
            
            obj.output_of_last_layer_mn = (obj.alpha_last_layer .* obj.prev_output_of_nn) + ((1.0 - obj.alpha_last_layer) .* obj.prev_output_of_mn);
            obj.input_to_last_layer_nn = (obj.weights_hidden_to_output_nn' * obj.output_of_hidden_layer_nn') + (obj.weights_hidden_to_output_mn' * obj.output_of_hidden_layer_mn') + (obj.beta .* obj.output_of_last_layer_mn');
            obj.output_nn = obj.output_layer_activation_function(obj.input_to_last_layer_nn);
            
            obj.prev_output_of_input_layer_nn = obj.output_of_input_layer_nn;
            obj.prev_output_of_input_layer_mn = obj.output_of_input_layer_mn;
            obj.prev_output_of_hidden_layer_nn = obj.output_of_hidden_layer_nn;
            obj.prev_output_of_hidden_layer_mn = obj.output_of_hidden_layer_mn;
            obj.prev_output_of_nn = obj.output_nn;
            obj.prev_output_of_mn = obj.output_of_last_layer_mn;
            output = obj.output_nn;
        end

        function backprop(obj, y_des)
            obj.y_des = y_des;
            obj.squared_error = mean((obj.output_nn - obj.y_des).^2);
            obj.rmse = sqrt(obj.squared_error);

            obj.error_last_layer = (obj.output_nn - obj.y_des) .* obj.output_layer_activation_function_derivative(obj.input_to_last_layer_nn);
            obj.error_hidden_layer = obj.activation_function_derivative(obj.input_to_hidden_layer_nn) .* (obj.weights_hidden_to_output_nn * obj.error_last_layer);

            obj.weights_hidden_to_output_nn = obj.weights_hidden_to_output_nn - (obj.neeta * repmat(obj.error_last_layer, 1, obj.number_of_hidden_neurons) .* repmat(obj.output_of_hidden_layer_nn, obj.number_of_output_neurons, 1)');
            obj.weights_input_to_hidden_nn = obj.weights_input_to_hidden_nn - (obj.neeta * repmat(obj.error_hidden_layer, obj.number_of_input_neurons, 1) .* (repmat(obj.output_of_input_layer_nn, 1, obj.number_of_hidden_neurons))');
            obj.weights_hidden_to_output_mn = obj.weights_hidden_to_output_mn - (obj.neeta * repmat(obj.error_last_layer, 1, obj.number_of_hidden_neurons) .* repmat(obj.output_of_hidden_layer_mn, obj.number_of_output_neurons, 1)');
            obj.weights_input_to_hidden_mn = obj.weights_input_to_hidden_mn - (obj.neeta * repmat(obj.error_hidden_layer, obj.number_of_input_neurons, 1) .* (repmat(obj.output_of_input_layer_mn, 1, obj.number_of_hidden_neurons))');
            obj.beta = obj.beta - (obj.neeta_dash * obj.error_last_layer * obj.output_of_last_layer_mn);
            
            obj.pd_e_wrt_v_hidden_layer = (obj.weights_hidden_to_output_mn * obj.error_last_layer)';
            obj.pd_e_wrt_v_input_layer = (obj.weights_input_to_hidden_mn * obj.error_hidden_layer)';
            obj.pd_e_wrt_v_last_layer = obj.beta * obj.error_last_layer;
            obj.pd_v_wrt_alpha_hidden_layer = obj.prev_output_of_hidden_layer_nn - obj.prev_output_of_hidden_layer_mn;
            obj.pd_v_wrt_alpha_input_layer = obj.prev_output_of_input_layer_nn - obj.prev_output_of_input_layer_mn;
            obj.pd_v_wrt_alpha_last_layer = obj.prev_output_of_nn - obj.prev_output_of_mn;
            
            obj.alpha_hidden_layer = obj.alpha_hidden_layer - (obj.neeta_dash * obj.pd_e_wrt_v_hidden_layer .* obj.pd_v_wrt_alpha_hidden_layer);
            obj.alpha_input_layer = obj.alpha_input_layer - (obj.neeta_dash * obj.pd_e_wrt_v_input_layer .* obj.pd_v_wrt_alpha_input_layer);
            obj.alpha_last_layer = obj.alpha_last_layer - (obj.neeta_dash * obj.pd_e_wrt_v_last_layer .* obj.pd_v_wrt_alpha_last_layer);

            obj.alpha_hidden_layer = max(0.0, min(1.0, obj.alpha_hidden_layer));
            obj.alpha_input_layer = max(0.0, min(1.0, obj.alpha_input_layer));
            obj.alpha_last_layer = max(0.0, min(1.0, obj.alpha_last_layer));
            obj.beta = max(0.0, min(1.0, obj.beta));

            if obj.spectral_norm
                norm_whn = norm(obj.weights_hidden_to_output_mn, 2);
                obj.weights_input_to_hidden_mn = (obj.weights_input_to_hidden_mn / norm(obj.weights_input_to_hidden_mn, 2)) * (obj.lipschitz_norm^(1/2));
                obj.weights_input_to_hidden_nn = (obj.weights_input_to_hidden_nn / norm(obj.weights_input_to_hidden_nn, 2)) * (obj.lipschitz_norm^(1/2));
                obj.weights_hidden_to_output_mn = (obj.weights_hidden_to_output_mn / norm_whn) * (obj.lipschitz_norm^(1/2));
                obj.weights_hidden_to_output_nn = (obj.weights_hidden_to_output_nn / norm(obj.weights_hidden_to_output_nn, 2)) * (obj.lipschitz_norm^(1/2));
                obj.beta = (obj.beta / norm(obj.beta, 2)) * (obj.lipschitz_norm^(1/2));
            end
        end

        function g1_x = activation_function(obj, x)
            g1_x = 15 * tanh(x / 15);
        end

        function g2_x = output_layer_activation_function(obj, x)
            g2_x = x;
        end

        function g1_dash_x = activation_function_derivative(obj, x)
            g1_dash_x = 1.0 - (tanh(x / 15))^2;
        end

        function g2_dash_x = output_layer_activation_function_derivative(obj, x)
            g2_dash_x = 1.0;
        end
    end
end


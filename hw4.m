clear; clc;
%% hyperparameters
layer1_units = 16;
layer2_units = 16;

num_iter = 5000;
step_size = 0.001;

%% Create dataset
dataset = cell(2,1);
dataset{1} = [0, 0, 0, 0, 1, 1, 1, 1;
              0, 0, 1, 1, 0, 0, 1, 1;
              0, 1, 0, 1, 0, 1, 0, 1]';

dataset{2} = [0, 1, 1, 0, 1, 0, 0, 1]';

%% Initialize layers
layer1 = FClayer(layer1_units, "tanh"); % try "relu" or "sigmoid" too.
layer2 = FClayer(layer2_units, "tanh");
layer3 = FClayer(1, "linear");

loss_graph = zeros(num_iter, 1);
accuracy_graph = zeros(num_iter, 1);
for iter = 1:num_iter
    %% Forward calculation
    layer1.forward(dataset{1});
    layer2.forward(layer1.hidden);
    layer3.forward(layer2.hidden);

    %% Back propagation
    loss = log_sigmoid_cross_entropy(layer3.hidden, dataset{2});
    layer3.backward(loss.delta, ones(1, 1));
    layer2.backward(layer3.delta, layer3.kernel);
    layer1.backward(layer2.delta, layer2.kernel);

    %% Parameter update
    layer1.update(step_size, dataset{1});
    layer2.update(step_size, layer1.hidden);
    layer3.update(step_size, layer2.hidden);
    
    loss_graph(iter) = mean(loss.error);
    accuracy_graph(iter) = sum(cast(loss.logit > 0.5, 'double') == dataset{2}) / length(loss.logit);
    if ~mod(iter, 50)
        sprintf('training loss: %f', loss_graph(iter))
        sprintf('training accuracy: %f%%', accuracy_graph(iter) * 100)
    end
end

plot(1:num_iter, loss_graph)
hold on
plot(1:num_iter, accuracy_graph)
legend('training error', 'training accuracy')
xlabel('iteration(s)')

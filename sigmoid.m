function y = sigmoid(x)
    % SHPark, SPA lab, Hanyang University. 2018.
    y = 1 ./ (1 + exp(-x));
end
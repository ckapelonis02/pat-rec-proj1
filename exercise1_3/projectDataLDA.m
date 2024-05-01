function [Z] = projectDataLDA(X, v)

% You need to return the following variables correctly.
Z = zeros(size(X, 1), 1);

Z = Z + X*v;

end

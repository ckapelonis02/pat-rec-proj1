function [X_rec] = recoverDataLDA(Z, v)

X_rec = zeros(size(Z, 1), length(v));

X_rec = X_rec + Z*v.';

end

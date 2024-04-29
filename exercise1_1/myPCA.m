function [eigenval, eigenvec, order] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [ eigenval, eigenvec, order] = mypca(X) computes eigenvectors of the autocorrelation matrix of X
%   Returns the eigenvectors, the eigenvalues (on diagonal) and the order 
%

% Useful values
[nSamples, nFeat] = size(X);

% Make sure each feature from the data is zero mean
[X_centered, mu, sigma] = featureNormalize(X);

% ====================== YOUR CODE HERE ======================
%

%FIX!!!!!!!!!!!
R = cov(X_centered); % Estimated covariance from samples

%D: diagonal matrix of eigenvalues
%V: Matrix columns are the eigenvectors
[V, D] = eig(R);

eigenval = diag(D); %Vector of eigenvalues
[eigenval, order] = sort(eigenval, 1, 'descend'); %Sort them
eigenvec = V(:, order); %Corresponding eigenvectors
varperc = eigenval / sum(eigenval); %Variance Contribution

% =========================================================================

end

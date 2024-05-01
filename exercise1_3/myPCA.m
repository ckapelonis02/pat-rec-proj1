function [U, S] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = principalComponentAnalysis(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[nSamples, nFeat] = size(X);

% You need to return the following variables correctly.
U = zeros(nFeat);
S = zeros(nFeat);

% ====================== YOUR CODE GOES HERE ======================
% Instructions: You should first compute the covariance matrix. Then, 
%  compute the eigenvectors and eigenvalues of the covariance matrix. 
%
% Note that the dataset X is normalized, when calculating the covariance


R = 1/nSamples .* transpose(X) * X;

%D: diagonal matrix of eigenvalues
%V: Matrix columns are the eigenvectors
[V, D] = eig(R);

S = diag(D); %Vector of eigenvalues
[S, order] = sort(S, 1, 'descend'); %Sort them
U = V(:, order); %Corresponding eigenvectors

% =========================================================================

end

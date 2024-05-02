%% Pattern Recognition
%  Exercise | Principle Component Analysis
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     myPCA.m
%     projectData.m
%     recoverData.m
%
%  Add code where needed.
%

  %% Initialization
clear all; close all; clc

%% ================== Part 1: Load Example Dataset  ===================
%  We start this exercise by using a small dataset that is easily to
%  visualize
%

%  The following command loads the breast cancert dataset. You should now have the 
%  variable X in your environment
Data = csvread('data/breast_cancer_data.csv');
NSamples = 100; %Get the first NSamples only for better visualization
X=Data(1:NSamples,1:end-1); % Get all features
Y=Data(1:NSamples,end);

fprintf('Visualizing example dataset for PCA.\n\n');
% Visualize the example dataset in 2D using sample feature pairs
% Repeat using different feature pairs on the 2D space
for i = 1:6
  subplot(2, 3, i);
  plot(X(Y==0, i), X(Y==0, i+6), 'bo',X(Y==1, i), X(Y==1, i+6), 'ro' );
  axis square;
  title('Data Samples in 2D')
  xlabel(sprintf('Feature %d', i));
  ylabel(sprintf('Feature %d', i+6));
end


fprintf('Program paused. Press enter to continue.\n');
pause

%% =============== Part 2: Principal Component Analysis ===============
%  You should now implement PCA, a feature extraction algorithm. 
%  Complete the code in myPCA.m
%  Consider only 2D samples by taking the first two features of the dataset

fprintf('\nRunning PCA on example dataset taking the first 2 features.\n\n');
X=Data(1:NSamples, 1:2); % Get 2 first features

% Before running PCA, it is important to first normalize X
% Add the necessary code to perform standardization in featureNormalize
[X_norm, ~, ~] = featureNormalize(X);

% Plotting before and after standardization
figure;
plot(X(Y==0, 1), X(Y==0, 2), 'bo',X(Y==1, 1), X(Y==1, 2), 'ro' );
axis square;
title('Data Samples in 2D before and after standardization')
xlabel('Feature 1');
ylabel('Feature 2');
hold on
plot(X_norm(Y==0, 1), X_norm(Y==0, 2), 'go',X_norm(Y==1, 1), X_norm(Y==1, 2), 'ko' );

%  Run PCA
[eigvals, eigvecs, ~] = myPCA(X_norm);

% Plot the samples on the first two principal components
figure;
plot(X_norm(Y==0, 1), X_norm(Y==0, 2), 'bo', X_norm(Y==1, 1), X_norm(Y==1, 2), 'ro');
hold on
% Plot the principal component vectors (arrows)
scale = 3; % Scale factor for the arrows
quiver(0, 0, eigvecs(1,1)*scale*eigvals(1), eigvecs(2,1)*scale*eigvals(1), 'r', 'LineWidth', 2, 'MaxHeadSize', 1);
quiver(0, 0, eigvecs(1,2)*scale*eigvals(2), eigvecs(2,2)*scale*eigvals(2), 'b', 'LineWidth', 2, 'MaxHeadSize', 1);
title('Normalized Samples and principal components')
xlabel('Normalized feature 1')
ylabel('Normalized feature 2')
axis square
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

% Calculate the Explained Variance
ExplainedVar = eigvals/sum(eigvals);
fprintf('Explained Variance (1st PC = %f) (2nd PC = %f)\n', ExplainedVar(1), ExplainedVar(2));

% Projection of the data onto the principal components
X_PCA = projectData(X_norm, eigvecs, 2);
X_PCA_1 = X_PCA(:, 1)*(eigvecs(:, 1).');
X_PCA_2 = X_PCA(:, 2)*(eigvecs(:, 2).');
% Plot the projection of the data onto the principal components
figure;
hold on
axis square;
plot(X_PCA_1(:, 1), X_PCA_1(:, 2), 'bo','MarkerFaceColor', 'b', 'MarkerEdgeColor', 'none', 'MarkerSize', 4 );
plot(X_PCA_2(:, 1), X_PCA_2(:, 2), 'ro','MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none', 'MarkerSize', 4 );
title('PCA of Breast Cancer Dataset');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
hold off;

% Print the first principal component
fprintf(' 1st Principal Component = %f %f \n', eigvecs(1,1), eigvecs(2,1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 3: Dimensionality Reduction ===================
%  Perform dimensionality reduction by projecting the data onto the 
%  first principal component. Then plot the data in this reduced in 
%  dimension space.  This will show you what the data looks like when 
%  using only the corresponding eigenvectors to reconstruct it.
%
%  You should complete the code in projectData.m
%
fprintf('\nDimensionality reduction on example dataset.\n\n');

%  Project the data onto K = 1 dimension
K = 1;
Z = projectData(X_norm, eigvecs, K);
fprintf('Projection of the first example: %f\n', Z(1));

%  Plot the data in this reduced dimension space
figure;
plot(Z(Y==0), Z(Y==0), 'bo', Z(Y==1), Z(Y==1), 'ro'); % eigvec is y=x line in 2D space
title('Reduced data');
xlabel('Values of data');

%  Use the K principal components to recover the Data in 2D
X_recover = recoverData(Z, eigvecs, K);
fprintf('Approximation of the first example: %f %f\n', X_recover(1, 1), X_recover(1, 2));

%  Draw the lines connecting the projected points to the original points
figure;
hold on;
plot(X_recover(Y==0, 1), X_recover(Y==0, 2), 'co','MarkerFaceColor', 'c', 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
plot(X_recover(Y==1, 1), X_recover(Y==1, 2), 'yo', 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'none', 'MarkerSize', 5);

plot(X_norm(Y==0, 1), X_norm(Y==0, 2), 'bo','MarkerFaceColor', 'b', 'MarkerEdgeColor', 'none', 'MarkerSize', 5 )
plot(X_norm(Y==1, 1), X_norm(Y==1, 2), 'ro','MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none', 'MarkerSize', 5 );
title('PCA sample projections');

axis square
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_recover(i,:), '--k', 'LineWidth', 1);
end
axis([-3 3.5 -3 3.5]); 
axis square
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =============== Part 4: Perform PCA on all features in the dataset =============
%  Apply PCA on the initial dataset and then choose the first 2 Principal Componets
%  to reduce the dimensionality of the dataset to the 2D space
%  The following code will load the dataset into your environment
%

X=Data(1:NSamples,1:end-1); % Get all features from the breast cancer dataset

% Before running PCA, it is important to first normalize X
% Add the necessary code to perform standardization in featureNormalize
[X_norm, ~, ~] = featureNormalize(X);

%  Run PCA
[eigvals, eigvecs, order] = myPCA(X_norm);

figure
hold on
% Plot the samples on the first two principal components
plot(X_norm(Y==0, order(1)), X_norm(Y==0, order(2)), 'bo','MarkerFaceColor', 'b', 'MarkerEdgeColor', 'none', 'MarkerSize', 5 )
plot(X_norm(Y==1, order(1)), X_norm(Y==1, order(2)), 'ro','MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none', 'MarkerSize', 5 );
title('PCA sample projections');
% Plot the principal component vectors (arrows)
scale = 1; % Scale factor for the arrows
quiver(0, 0, eigvecs(1,1)*scale*eigvals(1), eigvecs(2,1)*scale*eigvals(1), 'r', 'LineWidth', 2, 'MaxHeadSize', 1);
quiver(0, 0, eigvecs(1,2)*scale*eigvals(2), eigvecs(2,2)*scale*eigvals(2), 'b', 'LineWidth', 2, 'MaxHeadSize', 1);
hold off;
axis square;

% Projection of the data onto the principal components
% ADD YOUR CODE
K = 2;
X_PCA = projectData(X_norm, eigvecs, K);

% Projection of the data onto the principal components
X_PCA_1 = X_PCA(:, 1)*(eigvecs(:, 1).');
X_PCA_2 = X_PCA(:, 2)*(eigvecs(:, 2).');
% Plot the projection of the data onto the principal components
figure;
hold on
axis square;
plot(X_PCA_1(:, 1), X_PCA_1(:, 2), 'bo','MarkerFaceColor', 'b', 'MarkerEdgeColor', 'none', 'MarkerSize', 4 );
plot(X_PCA_2(:, 1), X_PCA_2(:, 2), 'ro','MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none', 'MarkerSize', 4 );
title('Breast Cancer Dataset - PCA reduced in 2D');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
hold off;

% Calculate the Explained Variance of the first 2 Principal Components
% ADD YOUR CODE
ExplainedVar = eigvals/sum(eigvals);
fprintf(' Explained Variance(1st PC = %f) (2nd PC = %f)\n', ExplainedVar(1), ExplainedVar(2));

% Print the first principal component
fprintf(' 1st Principal Component = %f %f \n', eigvecs(1,1), eigvecs(2,1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =============== Part 5: Loading and Visualizing Face Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  The following code will load the dataset into your environment
%
fprintf('\nLoading face dataset.\n\n');

%  Load Face dataset
load ('data/faces.mat')

%  Display the first 100 faces in the dataset
figure;
displayData(X(1:100, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 6: PCA on Face Data: Eigenfaces  ===================
%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  We display the first 36 eigenfaces.
%

fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this might take a minute or two ...)\n\n']);

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[eigvals, eigvecs, order] = myPCA(X_norm);

%  Visualize the top #numOfComps eigenvectors found (eigenfaces)
figure;
displayData(eigvecs(:, 1:36)');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============= Part 7: Dimensionality Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors 
%  If you are applying a machine learning algorithm 
fprintf('\nDimension reduction for face dataset.\n\n');

% Display normalized data
figure;
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

for K = [10, 50, 100, 200, 500]
    Z = projectData(X_norm, eigvecs, K);
    
    fprintf('The projected data Z has a size of: ')
    fprintf('%d ', size(Z));
    
    fprintf('\n\nProgram paused. Press enter to continue.\n');
    pause;
    
    %% ==== Part 8: Visualization of Faces after PCA Dimensionality Reduction ====
    %  Project images to the eigen space using the top K eigen vectors and 
    %  visualize only using those K dimensions
    %  Compare to the original input, which is also displayed
    
    fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');
    
    X_rec = recoverData(Z, eigvecs, K);
    
    % Display reconstructed data from only k eigenfaces
    figure;
    displayData(X_rec(1:100,:));
    title(sprintf('Recovered faces (%d principal compontents)', K));
    axis square;
end

%saving_figs('C:\Users\harilaos\Desktop\1\SMAP-proj1\exercise1_1\images');
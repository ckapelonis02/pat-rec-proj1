function A = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA

    [NumSamples NumFeatures] = size(Samples);

    A = zeros(NumFeatures, NewDim);
    
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels) then
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    
    Classes = unique(Labels);     %Return the unique elements of Labels
    NumClasses = length(Classes);  %The number of classes
    
    Sw = zeros(NumFeatures, NumFeatures);
    Sb = zeros(NumFeatures, NumFeatures);
    
    %Calculate the Global Mean
    m0 = mean(Samples);
    
    %For each class i
    %Find the necessary statistics
    for i = 1:NumClasses
      %Calculate the Class Prior Probability
      cl = Classes(i);
      P(i) = sum(Labels == cl) / NumLabels;
      
      %Calculate the Class Mean 
      mu(i, :) = mean(Samples(Labels == cl, :));
      
      %Calculate the Within Class Scatter Matrix
      Sw = Sw + P(i) * cov(Samples(Labels == cl, :));
      
      %Calculate the Between Class Scatter Matrix
      Sb = Sb + sum(Labels == cl) * (mu(i, :) - m0).' * (mu(i, :) - m0);
    end
  
    
    %Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = inv(Sw) * Sb;
    
    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
    if(NewDim > NumClasses-1) then
        fprintf('Illegal arguments.\n\n');
        exit
    end
    
    %Perform Eigendecomposition
    
    %D: diagonal matrix of eigenvalues
    %V: Matrix columns are the eigenvectors
    [V, D] = eig(EigMat);

    eigenval = diag(D); %Vector of eigenvalues
    [eigenval, order] = sort(eigenval, 1, 'descend'); %Sort them
    eigenvec = V(:, order); %Corresponding eigenvectors
    
    %% You need to return the following variable correctly.
    A = zeros(NumFeatures, NewDim);  % Return the LDA projection vectors
    A = A + eigenvec(:, 1:NewDim);   % Produce error if dimensions do not match
    
    
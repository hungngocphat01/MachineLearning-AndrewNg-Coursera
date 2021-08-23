function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

min_validation_err = 1e5;

Cs = [0.01, 0.03, 0.1, 0.3, 1, 3];
sigmas = [0.1 0.5 1 2];

C = 0
sigma = 0
for i=1:length(Cs)
    for j=1:length(sigmas)
        % Train SVM 
        fprintf("Training with C=%f, sigma=%f\n", Cs(i), sigmas(j));
        krnl = @(x1, x2) gaussianKernel(x1, x2, sigmas(j));
        trained_model = svmTrain(X, y, Cs(i), krnl);
        % Predict on cross validation set
        pred = svmPredict(trained_model, Xval);
        err = mean(double(pred ~= yval));
        % Regular find min algorithm
        if err < min_validation_err
            fprintf("Found new minima: %f\n", err);
            C = Cs(i);
            sigma = sigmas(j);
            min_validation_err = err;
        endif
    endfor
endfor

fprintf("Trained successfully with C=%f and sigma=%f", C, sigma)
% =========================================================================

end

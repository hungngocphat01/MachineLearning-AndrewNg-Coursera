function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));
  
  m = columns(X);
  for i = 1:m
    feature_col = X(:,i);
    mu(i) = mean(feature_col);
    sigma(i) = std(feature_col);
    X_norm(:,i) = (feature_col - mu(i))/sigma(i);
  endfor;
end

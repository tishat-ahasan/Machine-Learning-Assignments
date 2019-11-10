function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



h = X*theta;
a = (h-y).^2;
left = (1/(2*m))*a;
left = sum(left);
thetatmp = theta;

thetatmp(1,1)=0;

right = (sum(thetatmp.^2)*lambda)/(2*m);
J = left+right;



c = (h-y).*X;
c = sum(c);
l = (1/m) * c';
r = (lambda/m)*thetatmp;
grad = l+r;










% =========================================================================

grad = grad(:);

end

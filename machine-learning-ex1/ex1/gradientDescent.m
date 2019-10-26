function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
  
    h=X*theta;
    tmp1 = (h-y);
    tmpt0 = (sum(tmp1)*alpha)/m;
    tmp2 = tmp1.*X(:,2);
    tmpt1 = (sum(tmp2)*alpha)/m;
    tmpt0 = theta(1,1)-tmpt0;
    tmpt1 = theta(2,1)-tmpt1;
    theta(1,1) = tmpt0;
    theta(2,1) = tmpt1;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

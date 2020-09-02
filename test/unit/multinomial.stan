data {
  int N; //the number of training observations
  int N2; //the number of test observations
  int D; //the number of features
  int K; //the number of classes
  int y[N]; //the response
  matrix[N,D] x; //the model matrix
  matrix[N2,D] x_new; //the matrix for the predicted values
}
parameters {
  real alpha;
  matrix[D,K] beta; //the regression parameters
}

model {
  matrix[N, K] x_beta = x * beta;
  to_vector(beta) ~ normal(0, 5);

  for (n in 1:N)
    y[n] ~ categorical_logit(x_beta[n]');

}

generated quantities {
  vector[N2] y_new;
  matrix[N2, K] x_beta_new = x_new * beta;
  for (n in 1:N2)
   y_new[n] = categorical_rng(softmax(x_beta_new[n]')); //the y values predicted by the model
}

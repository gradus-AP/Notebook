//Time-invariant Gaussian State Space Model
//z[] : observed var. with values in real[obs_dim]
//x[] : state var. with values in real[state_dim]
//@transition equation
//x[t+1] = F * x[t] + H * W[t], W[t] ~ i.i.d. Normal(O, W)
//@observation equation
//z[t] = G * x[t] + V[t], V[t] ~ i.i.d. Normal(O, V) 
//@initial distribution
//x_0 ~ Normal(x_ini, Q)_init
functions {
  vector get_one_step_ahead_pred_state(matrix F, vector x){
    return F * x;
  }

  matrix get_one_step_ahead_pred_cov(matrix F, matrix W, matrix Q_filter, matrix H){
    return F * Q_filter * F' + H * W * H';
  }

  matrix get_Kalman_gain(matrix Q_pred, matrix G, matrix V) {
    return Q_pred * G' * inverse(G * Q_pred * G' + V);
  }

  vector get_filter_state(vector x_pred, vector z, matrix G, matrix Kalman_gain) {
    return x_pred + Kalman_gain * (z - G * x_pred);
  }  

  matrix get_filter_cov(int state_dim,matrix G, matrix Q_pred, matrix Kalman_gain) {
    return (diag_matrix(rep_vector(1, state_dim)) - Kalman_gain * G) * Q_pred ;
  }

  real get_likelihood(vector z, matrix V, vector x_pred, matrix Q_pred, matrix G){
    return multi_normal_lpdf(z | G * x_pred, G * Q_pred * G' + V);
  }  
}

data {
  int T;
  int state_dim;
  int obs_dim;
  int l;

  vector[state_dim] x_init;
  matrix[state_dim, state_dim] Q_init;

  vector[obs_dim] z[T];
  matrix[l,l] W;
  matrix[obs_dim, obs_dim] V;
  matrix[state_dim, l] H;
}

parameters {
  //AR parameter
  vector[state_dim] a;
}

transformed parameters {
  //model parameter
  matrix[state_dim,state_dim] F;
  matrix[obs_dim, state_dim] G;
  
  //one step ahead prediction distribution
  //x[t] | z[1:t-1] ~ Normal(x_pred[t], Q_pred[t])
  vector[state_dim] x_pred[T + 1];
  matrix[state_dim, state_dim] Q_pred[T + 1];
  
  //filter distribution
  //x[t] | z[1:t] ~ Normal(x_filter[t], Q_filter[t])
  vector[state_dim] x_filter[T];
  matrix[state_dim, state_dim] Q_filter[T];

  //Kalman gain
  matrix[state_dim, obs_dim] K[T];

  for (i in 1:(state_dim - 1) ) {
    for (j in 1:state_dim){
      if (j == i + 1) {
        F[i,j] = 1;
      } else {
        F[i,j] = 0;
      }
    }
  }
  
  for (j in 1:state_dim){
    F[state_dim, j] = - a[state_dim + 1 - j];
  }

  for (i in 1:obs_dim) {
    for (j in 1:state_dim){
      if (j < state_dim) {
        G[i,j] = 0;
      } else {
        G[i,j] = 1;
      }
    }
  }

  x_pred[1] = x_init;
  Q_pred[1] = Q_init;
  
  for (t in 1:T) {
    //Kalman gain
    K[t] = get_Kalman_gain(Q_pred[t], G, V);
    
    //Kalman filter
    x_filter[t] = get_filter_state(x_pred[t], z[t], G, K[t]);
    Q_filter[t] = get_filter_cov(state_dim, G, Q_pred[t], K[t]);
    
    //one step ahead prediction
    x_pred[t + 1] = get_one_step_ahead_pred_state(F, x_filter[t]);
    Q_pred[t + 1] = get_one_step_ahead_pred_cov(F, W, Q_filter[t], H);
  }
}


model {
  for (t in 1:T) {
    target += get_likelihood(z[t], V, x_pred[t], Q_pred[t], G);
  }
}
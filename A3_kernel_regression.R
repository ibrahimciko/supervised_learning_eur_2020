load("Airline.RData")
library(purrr)
#data
y <- scale(as.vector(Airline$output)) #standardize & demean
X <- scale(as.matrix(subset(Airline, select = -c(output)))) #standardize & demean
y.resp <-  as.vector(y)

calculate_kernels <- function(X, kernel,gamma = NULL, d = NULL){
  N <- dim(X)[1] #number of rows (observations)
  if ( kernel == "lin"){
    K = X%*%t(X)  ## Kernel matrix with nxn dimension
  } else if (kernel == "rbf"){
    K <- exp(-gamma * as.matrix(dist(X))^2)
  } else {
    K <- matrix(nrow = N, ncol = N)
    for (i in seq_len(N)){
      for (j in seq_len(N)){
        K[i,j] <- (1 + (X[i,]) %*% (X[j,]))^d
      }
    }
  }
  return(K)
}
#check
K <- calculate_kernels(X, kernel = "rbf", gamma = 0.2)
K <- calculate_kernels(X, kernel = "lin")
K <- calculate_kernels(X, kernel = "anything", d = 2)

#prediction kernel calculation
calculate_pred_kernels <- function(X, X_u,kernel, gamma = NULL, d = NULL){
  if (kernel == "lin"){
   K_u <-  X_u %*% t(X)
  } else { #either rbf or non_homogenous
    N <- dim(X)[1]
    N_u <- dim(X_u)[1] #number of rows in X_u
    K_u <- matrix(nrow =  N_u, ncol = N) # Kernel between X and X_u
    
    if (kernel =="rbf"){
      for( i in seq_len(N_u)){
        for (j in seq_len(N)){
          K_u[i,j] <- exp(-gamma*((X_u[i,]-X[j,]) %*% (X_u[i,]-X[j,])))
        }
      }
    } else {
        for( i in seq_len(N_u)){
          for (j in seq_len(N)){
            K_u[i,j] <- (1 + (X_u[i,]) %*% (X[j,]))^d
        }
      } 
    }
  }
  return(K_u)
}
#check whether code is working
#x_test <- matrix(1:10, ncol = 5)
#K_u <- calculate_pred_kernels(X,X_u = x_test, kernel = "rbf",gamma =0.2 )
#K_u <- calculate_pred_kernels(X,X_u = x_test, kernel = "lin")
#K_u <- calculate_pred_kernels(X,X_u = x_test, kernel = "poly",d = 2 )

cross_v <- function(X,y, k = 10, seed = 12345,kernel,gamma = NULL, d = NULL, lambda= NULL){
  set.seed(seed)
  shuffle_index <- sample(nrow(X))  #shuffle the index 
  X <- X[shuffle_index,]       #shuffled X
  y <- y[shuffle_index,]       #shuffled y
  folds_indices <- cut(seq(1,nrow(X)),breaks=k,labels=FALSE) #equally sized groups based on shuffled indices
  cv_err <- vector(mode = "double", length = k) #initialize cross_v error vector to keep the record of errors
  
  for (i in seq_len(k)){
    test_indices <- which(folds_indices == i, arr.ind = TRUE)  
    y_test <- y[test_indices];    X_test <- X[test_indices,]
    y_train <- y[-test_indices];  X_train <- X[-test_indices,]
    
    K<- calculate_kernels(X = X_train, kernel = kernel,
                          gamma = gamma,d = d )
    N <- dim(X_train)[1]
    a_hat <- solve(K + lambda * diag(N)) %*% y_train
    
    K_u <- calculate_pred_kernels(X = X_train, X_u = X_test, kernel = kernel, gamma = gamma,d = d)
    
    y_test_pred <- K_u %*% a_hat
    MSE_i <- t(y_test - y_test_pred) %*% (y_test - y_test_pred)/length(y_test) #error in the test set
    cv_err[i] <- MSE_i
  }
  return(sqrt(mean(cv_err)))
}
#check whether code is working
#cv_err <- cross_v(X = X, y = y, k = 10,seed = 12345,kernel = "rbf",gamma = 1/dim(X)[2],lambda = 0.01)
#cv_err <- cross_v(X = X, y = y, k = 10,seed = 12345,kernel = "lin",lambda = 0.5)
#cv_err <- cross_v(X = X, y = y, k = 10,seed = 12345,kernel = "any",d = 2, lambda = 0.5)

find_best_param <- function(X,y,k = 10,seed = 12345,kernel,gamma = NULL, lambda = NULL,d = NULL){
  #this function takes the parameters of the grid search for various kernel
  #it returns 2 outputs: a-) the tuple of best parameters and minimum loss, b-) all the losses of the grid
  #it at first creates a grid of parameters then uses the function cross_v defined above
  if(kernel == "lin"){
    grid <- expand.grid(lambda)
    names(grid) <- c("lambda")
    grid_n <- dim(grid)[1] #number of elements in the grid search
    loss_grid <- vector(mode = "list",length = grid_n)
    for (i in seq_len(grid_n)){
      avg_rmse = cross_v(X = X, y = y, k = k, seed = seed ,kernel,
                         gamma = NULL, lambda = grid$lambda[i],d = NULL)
      loss_grid[[i]] = c(avg_rmse,grid$lambda[i])
    }
  } else if(kernel == "rbf"){
    grid <- expand.grid(gamma,lambda)
    names(grid) <- c("gamma","lambda")
    grid_n <- dim(grid)[1] #number of elements in the grid search
    loss_grid <- vector(mode = "list",length = grid_n)
    for (i in seq_len(grid_n)){
      avg_rmse = cross_v(X = X, y = y, k = k, seed = seed ,kernel,
                         gamma = grid$gamma[i], lambda = grid$lambda[i],d = NULL)
      loss_grid[[i]] = c(avg_rmse,grid$gamma[i],grid$lambda[i])
    }
  } else {
    grid <- expand.grid(lambda,d)
    names(grid) <- c("lambda","d")
    grid_n <- dim(grid)[1] #number of elements in the grid search
    loss_grid <- vector(mode = "list",length = grid_n)
    for (i in seq_len(grid_n)){
      avg_rmse = cross_v(X = X, y = y, k = k, seed = seed ,kernel,
                         gamma = NULL, lambda = grid$lambda[i],d = grid$d[i])
      loss_grid[[i]] = c(avg_rmse,grid$lambda[i],grid$d[i])
    }
  }
  losses <- unlist(map(loss_grid,1))
  min_index <- which(losses == min(losses)) #return the index of lowest error
  tuple_of_lowest <- loss_grid[[min_index]]
  names(tuple_of_lowest) <- c("rmse_error",names(grid))
  output <- list("best_stat" = tuple_of_lowest,"all_losses" = loss_grid) #final output of the function
  names(output) <- 
  return(output)
}
res_rbf <- find_best_param(X = X, y = y, k = 10, seed = 12345, kernel = "rbf",
                gamma =  10^seq(-2, 6, length = 20),lambda = 10^seq(-2, 6, length = 10))
res_rbf$best_stat

res_lin <- find_best_param(X = X, y = y, k = 10, seed = 12345, kernel = "lin",
                      lambda = 10^seq(-2, 6, length = 10))
res_rbf$best_stat

res_poly <- find_best_param(X = X, y = y, k = 10, seed = 12345, kernel = "poly",
                       lambda = 10^seq(-2, 6, length = 10),d = c(2))
res_poly$best_stat


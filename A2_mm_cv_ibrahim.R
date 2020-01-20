library(purrr) # to use the map function to extract the first element of the list
load("supermarket1996.RData")  #load the data

df <- subset(supermarket1996,select = -c(STORE,CITY,ZIP,GROCCOUP_sum,SHPINDX)) #exclude the irrelevant var.
df <-na.omit(df) #omit the NA rows

y <- as.vector(df$GROCERY_sum)   # y variable
y <- scale(y) # scale and mean-center y

X <- scale(df[,2:dim(df)[2]]) #scale and mean-center y
X <- cbind(1,X) #add the column of 111111s to the X

calculate_elastic_loss <- function(X,y,beta,lambda,alpha){
  #This function calculates the loss of the target function given the parameters.
  #Note that it takes X which has already the intercept term
  n <- dim(X)[1]  #number of observation
  loss <- (2*n)^-1 * t((y - X%*%beta)) %*% (y - X%*%beta) +
    lambda * ((1-alpha)/(2* t(beta[2:length(beta)])%*%beta[2:length(beta)]) + alpha * sum(abs(beta[2:length(beta)])))
  return(loss)
}

mm_elasticnet <- function(X,y,is_standardized = TRUE, is_bias_added = FALSE, stop_crit = 10^-8, 
                   max_iter = 30,alpha, lambda, err_denominator = 10^-8){
  #------------------------mm_elasticnet finds the minimum of the loss function of a elastic net regression model
  #y --------------------- dependent variable of n observation
  #X --------------------- matrix of predictors with the shape n x p 
  #is_standardized -------a boolean to check whether the predictors are already standardized, default True
  #is_bias_added ---------a boolean to check whether the columns of 111s are in the X matrix
  #stop_crit -------------the floating value. Iteration stops if the change in the function is smaller than stop_crit
  #max_iter -------------- of iterations to be executed
  # alpha  --------------- weight of lasso component of the penalty. If 0, Ridge . IF 1, Lasso
  # lambda --------------- regularization penalty coefficient
  # err_denominator -----  floating value in the denominator of the surrogate function of g(x,z)
  
  if (is_standardized == FALSE){
    X <- scale(X)
  }
  
  if(is_bias_added == FALSE){
    X <- cbind(1,X)
  }
  p <- dim(X)[2]        # number of parameters including the constant
  n <- dim(X)[1]        # number of observation
  ident <- diag(p)      #identity matrix of dimension p x p
  ident_modified <- ident #modified identity matrix to not penalize the intercept coefficient
  ident_modified[1,1] <- 0 #modified identity matrix to not penalize the intercept coefficient
  beta  <- rnorm(p,mean = 0) #initialize the beta_s randomly
  
  #calculate the starting loss!
  loss <- calculate_elastic_loss(X = X, y = y, beta = beta, lambda = lambda, alpha = alpha)

  #----------- Loop -----------#
  iteration <- 0
  while(iteration < max_iter){
    #print(sprintf("starting iteration:%s  with loss: %s", iteration + 1, round(loss,2)))
    
    # the part where we need majorization sur_matrix = D, see the notes
    sur_matrix <- diag(1/as.vector(pmax(abs(beta),err_denominator)))
    
    sur_matrix[1,1] <- 0 #unpunish the intercept
    
    #A is the quadratic part of the loss function
    A <- n^(-1) * t(X)%*%X + lambda * (1 - alpha) * ident_modified + lambda * alpha * sur_matrix
    linear <- n^-1 *t(beta) %*% t(X)%*%y # the part where beta is linear
  
    # the constant (don't penalize the intercept)!
    c = (2*n)^-1* t(y)%*%y + 1/2 * lambda * alpha * sum(abs(beta[2:length(beta)]))
    

    surrogate_loss <- 1/2 * t(beta) %*% A %*% beta - linear + c  #surrogate loss g(x,z) in the notes
    beta_candid <- n^-1 *  solve(A, ident,tol = 1e-19) %*% t(X) %*% y  #candidate beta,
    
    #new loss of the original loss function
    loss_new <- calculate_elastic_loss(X = X, y = y, beta = beta_candid, lambda = lambda, alpha = alpha) 
    if(abs(loss_new - loss) < stop_crit | loss < loss_new){
      print(sprintf("Either Difference between losses are smaller than the stopping criterion: %s. Or
               previous loss was lower. Therefore iteration ends", stop_crit))
      loss_new <- loss # so do not update the loss_new
      break
    } else{
      print(sprintf("Loss decreased by %s. Iteration continues with updating the beta coefficients",  loss - loss_new))
      loss <- loss_new #update the loss with the candidate beta
      beta <- beta_candid  #beta is updated as the candidate beta (algorithm continues to iterate)
    }
  }
  #create the result of the function
  res <- list(beta,alpha,lambda,iteration,loss)
  names(res) <- c("beta", "alpha","lambda","iteration","loss")
  return(res)
}

library(glmnet)
set.seed(12345)
my_result <- mm_elasticnet(X = X, y = y, is_standardized = TRUE, is_bias_added = TRUE,max_iter = 30, alpha = 0.5, lambda = 1)

##### K-fold cross validation
cross_v <- function(X,y,k = 5,seed = 12345,alpha,lambda){
  shuffle_index <- sample(nrow(X))  #shuffle the index 
  X <- X[shuffle_index,]       #shuffled X
  y <- y[shuffle_index,]       #shuffled y
  
  folds_indices <- cut(seq(1,nrow(X)),breaks=k,labels=FALSE) #equally sized groups based on shufflerd indices
  cv_err <- vector(mode = "double", length = k) #initialize cross_v error vector to keep the record of errors
  for (i in seq_len(k)){
    test_indices <- which(folds_indices == i, arr.ind = TRUE)  
    y_test <- y[test_indices] 
    X_test <- X[test_indices,]
    y_train <- y[-test_indices]
    X_train <- X[-test_indices,]
    fit <- mm_elasticnet(X = X_train, y = y_train, 
                         is_standardized = TRUE, is_bias_added = TRUE,
                         max_iter = 30, alpha = alpha, lambda = lambda)
    y_test_pred <- X_test %*% fit$beta #predicted y_test values
    MSE_i <- t(y_test - y_test_pred) %*% (y_test - y_test_pred)/length(y_test) #error in the test set
    cv_err[i] <- MSE_i
  }
  return(mean(cv_err))
}
cv_err <- cross_v(X = X, y = y, k = 5,alpha = 0.5, lambda = 1, seed = 12345) 

find_best_param <- function(X,y,k = 5,alpha, lambda,seed){
  #this function takes the vector of alpha and lambda which are the hyperparameters of elasticnet
  #it returns 2 outputs: a-) the tuple of lambda alpha and minimum loss, b-) all the losses of the grid
  # it at first creates a grid of parameters then uses the function cross_v defined above
  # at the end it selects the index of lowest error and return the corresponding alpha,lambda,loss values
  grid <- expand.grid(alpha,lambda)
  names(grid) <- c("alpha","lambda")
  grid_n <- dim(grid)[1] #number of elements in the grid search
  
  loss_grid <- vector(mode = "list",length = grid_n)
  
  for (i in seq_len(grid_n)){
    avg_mse = cross_v(X = X, y = y, k = k, seed = seed ,alpha = grid$alpha[i],lambda = grid$lambda[i])
    loss_grid[[i]] = c(avg_mse,grid$alpha[i],grid$lambda[i])
  }
  losses <- unlist(map(loss_grid,1))
  min_index <- which(losses == min(losses)) #return the index of lowest error
  tuple_of_lowest <- loss_grid[[min_index]]
  names(tuple_of_lowest) <- c("error","alpha","lambda")
  output <- list(tuple_of_lowest,loss_grid) #final output of the function
  return(output)
}
#result ridge
res <- find_best_param(X = X,y = y, k = 10, alpha = c(0),
                lambda = 10^seq(-2, 5, length = 10), seed = 12345)
#result lasso
res <- find_best_param(X = X,y = y, k = 10, alpha = c(1),
                       lambda = 10^seq(-2, 5, length = 10), seed = 12345)


#calculating the sum of squared residuals with the optimal paramters
res[1]
res_correct_param <- mm_elasticnet(X = X, y = y, is_standardized = TRUE, is_bias_added = TRUE
              ,max_iter = 30, alpha = res[[1]][2], lambda = res[[1]][3])

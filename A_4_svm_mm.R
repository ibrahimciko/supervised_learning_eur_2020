## Prepare Heart data for SVM
data("Heart", package = "dsmle")  
Heart <- na.omit(Heart)
y <- Heart$AHD
X <- model.matrix(AHD ~ ., data = Heart)

sample_divider <- function(X,y){
  #This function separates the y = +1 and y = -1, samples for further calculations
  # subscript "p" denotes the positive examples whereas subscript "n" the negatives
  # y can be a string input
  levels(y) <- c(-1 ,1)
  ind_p <- which(y == 1) 
  y_p <- y[ind_p]; X_p <- X[ind_p,]
  y_n <- y[-ind_p]; X_n <- X[-ind_p,]
  result <- list(y_p,X_p,y_n,X_n,ind_p)
  names(result) <- c("y_p","X_p","y_n","X_n","ind_p")
  return(result)
}


preprocesser <- function(X,mean = NULL,sd = NULL,with_bias = T, standardized = NULL,is_train){
  #This function cleans the bias and standardize the X matrix:
  #If the target object is a Test set, it standardize based on a given Mean and SD
  #If the target object is a Train set,it standardize and records the mean and SD
  if (with_bias == T){
    X <- X[,2:dim(X)[2]] 
  }
  if (is_train == T) {
    if(standardized == F){
    X <- scale(X)
    mean <- attr(X, 'scaled:center')
    sd <- attr(X,'scaled:scale')
    scale_param <- list(mean,sd)
    names(scale_param) <- c("mean","sd")
    res <- list(X, scale_param)
    names(res) <- c("X","scale_param")
    }
  } else {
    res <-  scale(X, center = mean, scale = sd)
  }
  return(res)
}

calculate_kernels <- function(X, kernel = "linear",sigma = NULL, d = NULL){
  N <- dim(X)[1] #number of rows (observations)
  if ( kernel == "linear" & dim(X)[2] > dim(X)[1]){
    X = tcrossprod(X)  ## Kernel matrix with nxn dimension
  } else if (kernel == "rbf"){
    X <- exp(-(1/sigma) * as.matrix(dist(X))^2)
  } else if (kernel =="poly") {
    X <- ( tcrossprod(X) + 1) ^ d
  } 
  return(X)
}

calculate_pred_kernels <- function(X = NULL, X_u, kernel, sigma = NULL, d = NULL){
  if (kernel == "linear" & dim(X)[2] > dim(X)[1]){
    K_u <- X_u
  } else { #either rbf or non_homogenous
    N <- dim(X)[1]
    N_u <- dim(X_u)[1] #number of rows in X_u
    K_u <- matrix(nrow =  N_u, ncol = N) # Kernel between X and X_u
    
    if (kernel =="rbf"){
      for( i in seq_len(N_u)){
        for (j in seq_len(N)){
          K_u[i,j] <- exp(-sigma*((X_u[i,]-X[j,]) %*% (X_u[i,]-X[j,])))
        }
      }
    } else if (kernel =="poly") {
      for( i in seq_len(N_u)){
        for (j in seq_len(N)){
          K_u[i,j] <- (1 + (X_u[i,]) %*% (X[j,]))^d
        }
      } 
    }
  }
  return(K_u)
}

predict_confusion <- function(X,y,w){
  levels(y) <- c(-1,1)
  y_hat <- rep(NA ,length = length(y))
  q <-  X %*% w
  y_hat <- ifelse(q >= 0, 1 , -1)
  sensitivity <- sum(y == 1 & y_hat == 1) / sum( y == 1)
  precision <- sum(sum(y == 1 & y_hat == 1))/sum(y_hat == 1)
  hitrate <- sum(y == y_hat)/ length(y)
  specificity <- 1 - (sum(y == -1 & y_hat == 1)) / sum(y == -1)
  res <- list(q,y_hat,sensitivity,precision,hitrate,specificity)
  names(res) <- c("q","y_hat","sensitivity","precision","hitrate","specificity")
  return(res)
}

loss_svm <- function(X_p,X_n,y_n,y_p,w,lambda,hinge = "absolute"){
  #This function calculates the loss of the SVM.
  #Note that it takes X which has already the intercept term
  q_p <- X_p %*% w
  q_n <- X_n %*% w 
  penalty <- lambda * t(w[2:length(w)]) %*% w[2:length(w)]  # exclude the intercept
  
  if (hinge == "absolute"){
    loss_p <- sum(pmax(0, 1 - q_p ))
    loss_n <- sum(pmax(0, 1 + q_n)) 
    loss <- loss_p + loss_n
  } else if (hinge == "quadratic"){
    loss_p <- sum(pmax(0,1 - q_p)^2)
    loss_n <- sum(pmax(0,1 + q_n)^2)
    loss <- loss_p + loss_n
  }
  total_loss <- loss + penalty
  return(total_loss)
}

g_loss <- function(X_p,X_n,y_n,y_p,w,lambda,epsilon = 1e-8,hinge = "absolute"){
  p <- length(w) #number of variables
  q_n <- drop(X_n %*% w)  #positive predictions
  q_p <- drop(X_p %*% w)  #negative predictions
  n_n <- length(y_n) # number of negative examples
  n_p <- length(y_p) # number of positive examples
  if (hinge == "absolute"){
    #majorizing the negative labels
    a_n <- (4 * pmax(epsilon,abs(q_n + 1)))^-1  #vector a_n with negative labels
    b_n <- -(a_n + 1/4)
    c_n <- a_n + 1/2 + abs(q_n + 1)/4
    #majorizing the positive labels
    a_p <-  (4 * pmax(epsilon,abs(1 - q_p)))^-1 #vector a_p with positive labels
    b_p <- a_p + 1/4
    c_p <- a_p + 1/2 + abs(1 - q_p)/4
  } else if (hinge == "quadratic"){
    #majorizing the negative labels
    a_n <- rep(1, n_n)
    q_n_smaller_ind <- which(q_n <= -1) # majorizing condition
    
    b_n <- vector(mode = "double", length = n_n)
    b_n[q_n_smaller_ind ] <- q_n[q_n_smaller_ind ]
    b_n[-q_n_smaller_ind] <- -1
    
    c_n <- vector(mode = "double", length = n_n)
    c_n[q_n_smaller_ind ] <- 1 - 2 * (q_n[q_n_smaller_ind ] + 1) + (q_n[q_n_smaller_ind] +1)^2
    c_n[-q_n_smaller_ind] <- 1
    
    #majorizing the positive labels
    a_p <- rep(1, n_p)
    q_p_smaller_ind <- which(q_p <= 1)
    
    b_p <- vector(mode = "double", length = n_p)
    b_p[q_p_smaller_ind] <- 1 
    b_p[-q_p_smaller_ind] <- q_p[-q_p_smaller_ind]
    
    c_p <- vector(mode = "double", length = n_p)
    c_p[q_p_smaller_ind] <- 1
    c_p[-q_p_smaller_ind] <- 1 - 2*(1 - q_p[-q_p_smaller_ind]) + ( 1- q_p[-q_p_smaller_ind])^2
  }
  P <- diag(p); P[1,1] <- 0 # unpenalize the intercept
  A <- diag(c(a_n, a_p))
  X <- rbind(X_n, X_p)
  b <- c(b_n, b_p)
  c <- c(c_n, c_p)
  K <- (t(X) %*% A) %*% X + lambda * P
  loss <- t(w) %*% K %*% w - 2 * t(w)%*%t(X) %*% b  + sum(c)
  
  #update w
  w <- solve(K) %*% t(X) %*% b
  
  #out
  out <- list(loss,w)
  names(out) <- c("gloss","w")
  return(out)
}

mm_svm <- function(X,y,standardized = TRUE, with_bias = TRUE, lambda, hinge = "absolute",
                   kernel = NULL, sigma = NULL, d = NULL,epsilon = 1e-8,max_iter = 30){
  #y --------------------- dependent variable of n observation
  #X --------------------- matrix of predictors with the shape n x p 
  #standardized -------a boolean to check whether the predictors are already standardized, default True
  #with_bias --------- 1 if the X contains the columns of 1s as the bias  
  #epsilon -------------the floating value. Iteration stops if the change in the function is smaller than epsilon
  #max_iter -------------- of iterations to be executed
  # lambda --------------- regularization penalty coefficient
  # kernel --------------- kernelized svm: options: "rbf(sigma)", "poly (set d)", "linear". NULL ----> no kernelization
  processed <- preprocesser(X,with_bias = T, standardized = F,is_train = T)
  X <- processed$X; scale_param <- processed$scale_param
    
  X <- calculate_kernels(X, kernel, sigma,d) #decide whether to Kernel or not
  X <- cbind(1,X) #add bias
  
  #divide the positive and negative examples
  labels <- sample_divider(X = X,y = y)
  X_p <- labels$X_p; y_p <- labels$y_p
  X_n <- labels$X_n; y_n <- labels$y_n
  ind_p <- labels$ind_p
  
  w <- rnorm(dim(X)[2]) #initialize the weight parameter
  
  loss_f <- loss_svm(X_p,X_n,y_n,y_p,w,lambda,hinge) #calculate the starting loss!
  
  #----------- Loop -----------#
  iteration <- 0
  while(iteration < max_iter){
    iteration <- iteration + 1
    major <- g_loss(X_p,X_n,y_n,y_p,w,lambda,epsilon,hinge)
    gloss <- major$gloss #loss of the surrogate function g
    w_update <- major$w # candidate weight
    loss_update <- loss_svm(X_p,X_n,y_n,y_p,w_update,lambda,hinge) 
    if(abs(loss_update - loss_f) < epsilon | loss_f < loss_update){
      print(sprintf("Either Difference between losses are smaller than the stopping criterion: %s. Or
               previous loss was lower. Therefore iteration ends", epsilon))
      break
    } else{
      print(sprintf("Loss decreased by %s. Iteration continues with updating the beta coefficients",  loss_f - loss_update))
      loss_f <- loss_update #update the loss with the candidate beta
      w <- w_update  #beta is updated as the candidate beta (algorithm continues to iterate)
    }
  }
  #get the predictions
  confusion <-predict_confusion(X,y,w)
  #create the result of the function
  res <- list(w,lambda,iteration,loss_f,confusion,ind_p,scale_param)
  names(res) <- c("w","lambda","iteration","loss","confusion","ind_p","scale_param")
  return(res)
}
#linear - absolute
result <- mm_svm(X = X, y = y, standardized = F,with_bias = T,kernel = "linear",sigma = 1,
                 lambda = 1,epsilon = 1e-8,max_iter = 500,hinge = "absolute")
result$scale_param
result$confusion

standardized = F;with_bias = T;kernel = "linear";sigma = 1
lambda = 1 ; epsilon = 1e-8 ;max_iter = 500 ;hinge = "absolute"

#linear quadratic
result <- mm_svm(X = X, y = y, standardized = F,with_bias = T,kernel = "linear",sigma = 1,
                 lambda = 1,epsilon = 1e-4,max_iter = 500,hinge = "quadratic")
result$confusion
#standardized = F;with_bias = T;kernel = "linear";sigma = 1
#lambda = 1 ; epsilon = 1e-8 ;max_iter = 500 ;hinge = "quadratic"

#rbf quadratic
result <- mm_svm(X = X, y = y, standardized = F,with_bias = T,kernel = "rbf",sigma = 2,
                 lambda = 1,epsilon = 1e-4,max_iter = 500,hinge = "quadratic")
result$confusion

#rbf absolute
result <- mm_svm(X = X, y = y, standardized = F,with_bias = T,kernel = "rbf",sigma = 2,
                 lambda = 1,epsilon = 1e-4,max_iter = 500,hinge = "absolute")
result$confusion

##################### Cross Validation ################3
cross_v <- function(X,y,standardized,with_bias, k = 10, seed = 12345,
                    lambda= NULL,hinge,kernel,sigma = NULL, d = NULL, 
                    epsilon,max_iter){
  set.seed(seed)
  shuffle_index <- sample(nrow(X))  #shuffle the index 
  X <- X[shuffle_index,]       #shuffled X
  y <- y[shuffle_index]       #shuffled y
  folds_indices <- cut(seq(1,nrow(X)),breaks=k,labels=FALSE) #equally sized groups based on shuffled indices
  cv_hitrate <- vector(mode = "double", length = k) #initialize cross_v error vector to keep the record of errors
  
  for (i in seq_len(k)){
    test_indices <- which(folds_indices == i, arr.ind = TRUE)  
    y_test <- y[test_indices];    X_test <- X[test_indices,]
    y_train <- y[-test_indices];  X_train <- X[-test_indices,]
    
    fit <- mm_svm(X_train,y_train,standardized,with_bias,lambda,hinge,kernel,sigma,d,epsilon,max_iter)
    scale_param <- fit$scale_param
    X_test <- preprocesser(X_test, mean = scale_param$mean, sd = scale_param$sd,
                           with_bias = T,standardized = F,is_train = F)
    w <- fit$w
    
    if (kernel != "linear"){
      X_test <- calculate_pred_kernels(X_train,X_u = X_test, kernel, sigma,d)
    }
    #add bias
    X_test <- cbind(1,X_test)
    cv_hitrate[i] <- predict_confusion(X_test,y_test,w)$hitrate
  }
  #create the result of the function
  res <- list(cv_hitrate,lambda,hinge,kernel,sigma,d)
  names(res) <- c("cv_hitrate","lambda","hinge","kernel","sigma","d")
  return(res)
}
#####CV: kernel linear, hinge "absolute
res <- cross_v(X,y,standardized = F,with_bias = T, k = 5, seed = 12345,
       lambda = 1,hinge = "absolute",kernel = "linear",sigma = NULL, d = NULL, 
                    epsilon = 1e-4,max_iter = 100)
res


standardized = F;with_bias = T; k = 5; seed = 12345;
lambda = 1 ; hinge = "absolute" ; kernel = "linear" ;sigma = NULL ; d = NULL ;
epsilon = 1e-4 ;max_iter = 100

###CV: hinge = absolute , kernel "rbf
cross_v(X,y,standardized = F,with_bias = T, k = 5, seed = 12345,
        lambda = 1,hinge = "absolute",kernel = "rbf",sigma = 1, d = NULL, 
        epsilon = 1e-4,max_iter = 100)
standardized = F; with_bias = T ; k = 5 ; seed = 12345 ;
lambda = 1 ;hinge = "absolute" ;kernel = "rbf" ;sigma = 1; d = NULL;
epsilon = 1e-4;max_iter = 100

find_best_param <- function(X,y,k = 10,seed = 12345,kernel,sigma = NULL, 
                            lambda = NULL,d = NULL,standardized = F,max_iter = 100,
                            with_bias = T, hinge = "absolute", epsilon =1e-4){
  
  #this function takes the parameters of the grid search for various kernel
  #it returns 2 outputs: a-) the tuple of best parameters and minimum loss, b-) all the losses of the grid
  #it at first creates a grid of parameters then uses the function cross_v defined above
  if(kernel == "linear"){
    grid <- expand.grid(lambda)
    names(grid) <- c("lambda")
    grid_n <- dim(grid)[1] #number of elements in the grid search
    loss_grid <- vector(mode = "list",length = grid_n)
    for (i in seq_len(grid_n)){
      hitrate = mean(cross_v(X,y,standardized,with_bias,k, seed,lambda = grid[i,1],hinge,
                         kernel, sigma, d,epsilon,max_iter)$cv_hitrate)
      loss_grid[[i]] = c(hitrate,grid$lambda[i])
    }
  } else if(kernel == "rbf"){
    grid <- expand.grid(sigma,lambda)
    names(grid) <- c("sigma","lambda")
    grid_n <- dim(grid)[1] #number of elements in the grid search
    loss_grid <- vector(mode = "list",length = grid_n)
    for (i in seq_len(grid_n)){
      hitrate = mean(cross_v(X,y,standardized,with_bias,k, seed,lambda = grid[i,1],hinge,
                             kernel, sigma = grid[i,2], d,epsilon,max_iter)$cv_hitrate)
      loss_grid[[i]] = c(hitrate ,grid$lambda[i],grid$sigma[i])
    }
  } else if (kernel == "poly") {
    grid <- expand.grid(lambda,d)
    names(grid) <- c("lambda","d")
    grid_n <- dim(grid)[1] #number of elements in the grid search
    loss_grid <- vector(mode = "list",length = grid_n)
    for (i in seq_len(grid_n)){
      hitrate = mean(cross_v(X,y,standardized,with_bias,k, seed,lambda = grid[i,1],hinge,
                             kernel, sigma, d = grid[i,2] ,epsilon,max_iter)$cv_hitrate)
      loss_grid[[i]] = c(hitrate,grid$lambda[i],grid$d[i])
    }
  }
  return(loss_grid)
}

k = 5; seed = 12345; kernel ="linear"; sigma = NULL
lambda = 2^seq(5, -5, length.out = 19);d = NULL; standardized = F; max_iter = 100;
with_bias = T; hinge = "quadratic"; epsilon =1e-4

find_best_param(X,y,k = 5, seed =12345, kernel = "linear", sigma = NULL, d= NULL,
                lambda = 2^seq(5, -5, length.out = 19),epsilon = 1e-4,
                standardized = F,with_bias = T,max_iter = 100,hinge = "quadratic")
library(SVMMaj)      
svmmajcrossval(X,y,hinge = "absolute",)

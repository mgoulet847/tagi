#' Result of the TAGI with derivative calculations
#'
#' This function returns the resulting derivatives from the neural network model using TAGI.
#'
#' @param NN Lists the structure of the neural network
#' @param xtrain Training set of input variables
#' @param ytrain Training set of responses
#' @param xtest Testing set of input variables
#' @param ytest Testing set of responses
#' @return - Mean of predicted responses
#' @return - Variance of the predicted responses
#' @return - Mean of first derivative of predicted responses
#' @return - Mean of second derivative of predicted responses
#' @export
runBatchDerivative <- function(NN, xtrain, ytrain, xtest, ytest){
  # Initialization
  maxEpoch <- NN$maxEpoch

  # Train net
  out_initialization <- initialization(NN)
  NN <- out_initialization[[1]]
  states <- out_initialization[[2]]

  theta <- initializeWeightBiasD(NN)
  normStat <- createInitNormStat(NN)

  # Test net
  NNtest = NN
  NNtest$trainMode = 0
  NNtest$batchSize = 1
  NNtest$repBatchSize = 1

  out_initialization <- initialization(NNtest)
  NNtest <- out_initialization[[1]]
  statesT <- out_initialization[[2]]
  normStatT <- createInitNormStat(NNtest)

  # Training
  stop = 0
  epoch = 0
  Sxopt = 0.01
  Sx = Sxopt * rep(1, nrow(xtrain))

  while (stop == 0){
    epoch = epoch + 1
    if (epoch >= 1){
      idxtrain = sample(nrow(ytrain))
      ytrain = matrix(ytrain[idxtrain,], nrow = length(idxtrain))
      xtrain = matrix(xtrain[idxtrain,], nrow = length(idxtrain))
    }
    out_batchDerivative = batchDerivative(NN, theta, normStat, states, xtrain, Sx, ytrain, 1)
    theta = out_batchDerivative[[1]]
    normStat = out_batchDerivative[[2]]
    yptrain = out_batchDerivative[[3]]
    Syptrain = out_batchDerivative[[4]]
    dyptrain = out_batchDerivative[[5]]

    if(epoch >= NN$maxEpoch){
      stop = 1
    }
  }

  # Testing
  out_batchDerivative = batchDerivative(NNtest, theta, normStatT, statesT, xtest, Sx, NULL, 1)
  yp = out_batchDerivative[[3]]
  Syp = out_batchDerivative[[4]]
  dyp = out_batchDerivative[[5]]
  ddyp = out_batchDerivative[[8]]

  outputs <- list(yp, Syp, dyp, ddyp)
  return(outputs)
}

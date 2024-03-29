#' Regression problem
#'
#' This function trains neural network models to solve a regression problem.
#'
#' @param NN Lists the structure of the neural network
#' @param x Input data
#' @param y Response data
#' @param trainIdx Observations IDs that are assigned to the training set
#' @param testIdx Observations IDs that are assigned to the testing set
#' @return - Mean vector of parameters for each layer \eqn{\mu_{\theta}}
#' @return - Covariance matrix of parameters for each layer \eqn{\Sigma_{\theta}}
#' @return - RMSE and LL metrics for each network models created
#' @return - Training time of each neural network models created
#' @return - Mean of predicted responses
#' @return - Variance of the predicted responses
#' @export
regression <- function(NN, x, y, trainIdx, testIdx){
  # Initialization
  initsv = NN$sv
  initmaxEpoch = NN$maxEpoch
  NN$errorRateEval = 0

  # Indices for each parameter group
  # Train net
  NN$trainMode = 1
  NN$batchSize = NN$batchSizeList[1]
  NN <- initialization_net(NN)
  NN <- parameters(NN)
  NN <- covariance(NN)

  # Validation net
  NNval = NN
  NNval$trainMode = 0
  NNval$batchSize = NN$batchSizeList[2]
  NNval <- parameters(NNval)
  NNval <- covariance(NNval)

  # Test net
  NNtest = NN
  NNtest$trainMode = 0
  NNtest$batchSize = NN$batchSizeList[3]
  NNtest <- parameters(NNtest)
  NNtest <- covariance(NNtest)

  # Loop
  Nsplit = NN$numSplits
  RMSElist = rep(0, Nsplit)
  LLlist = rep(0, Nsplit)
  trainTimelist = rep(0, Nsplit)
  permuteData = 0

  if (is.null(trainIdx)||is.null(testIdx)){
    permuteData = 1
  }

  for (s in 1:Nsplit){
    old = Sys.time()
    if (permuteData == 1){
      out_split = split(x, y, NN$ratio)
      xtrain = out_split[[1]]
      ytrain = out_split[[2]]
      xtest = out_split[[3]]
      ytest = out_split[[4]]
    } else {
      xtrain = x[trainIdx[[s]],]
      ytrain = matrix(y[trainIdx[[s]],], ncol = NN$ny)
      xtest = x[testIdx[[s]],]
      ytest = matrix(y[testIdx[[s]],], ncol = NN$ny)
    }
    out_normalize = normalize(xtrain, ytrain, xtest, ytest)
    xtrain = out_normalize[[1]]
    ytrain = out_normalize[[2]]
    xtest = out_normalize[[3]]
    mytrain = out_normalize[[7]]
    sytrain = out_normalize[[8]]

    # Initialize weights & bias
    out_initializeWeightBias <- initializeWeightBias(NN)
    mp = out_initializeWeightBias[[1]]
    Sp = out_initializeWeightBias[[2]]

    # Training
    NN$trainMode = 1
    stop = 0
    epoch = 0

    while (stop == 0){
      if (epoch > 1){
        idxtrain = sample(nrow(ytrain))
        ytrain = matrix(ytrain[idxtrain,], nrow = length(idxtrain))
        xtrain = xtrain[idxtrain,]
      }
      epoch = epoch + 1
      out_network = network(NN, mp, Sp, xtrain, ytrain)
      mp = out_network[[1]]
      Sp = out_network[[2]]

      if(epoch >= NN$maxEpoch){
        stop = 1
      }
    }

    # Testing
    NNtest$sv = NN$sv
    out_network = network(NNtest, mp, Sp, xtest, NULL)
    ynTest = out_network[[3]]
    SynTest = out_network[[4]]
    R = matrix(NNtest$sv^2, nrow = nrow(SynTest), ncol = ncol(SynTest))
    SynTest = SynTest + R
    out_denormalize <- denormalize(ynTest, SynTest, mytrain, sytrain)
    ynTest = out_denormalize[[1]]
    SynTest = out_denormalize[[2]]

    # Evaluation
    RMSElist[s] = computeError(ytest, ynTest)
    LLlist[s] = loglik(ytest, ynTest, SynTest)
    new = Sys.time() - old
    trainTimelist[s] = new

    print("")
    cat(sprintf("Results for Run # %s, RMSE: %s and LL: %s", s, RMSElist[s], LLlist[s]))
  }
  metric = list("RMSElist" = RMSElist, "LLlist" = LLlist)
  outputs <- list(mp, Sp, metric, trainTimelist, ynTest, SynTest)
  return(outputs)
}

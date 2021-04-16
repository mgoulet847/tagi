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
    if (epoch >= 1){
      idxtrain = sample(nrow(ytrain))
      ytrain = matrix(ytrain[idxtrain,], nrow = length(ytrain[idxtrain,]))
      xtrain = xtrain[idxtrain,]
    }
    epoch = epoch + 1
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
}
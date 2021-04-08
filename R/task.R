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
}

runBatchDerivative <- function(NN, xtrain, ytrain, xtest, ytest){
  # Initialization
  maxEpoch <- NN.maxEpoch

  # Train net
  out_initialization <- initialization(NN)
  NN <- out_initialization[[1]]
  states <- out_initialization[[2]]
}

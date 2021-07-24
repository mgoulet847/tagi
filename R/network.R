#' One iteration of the TAGI with derivative calculations
#'
#' This function goes through one learning iteration of the neural network model
#' using TAGI with derivative calculations.
#'
#' @param NN Lists the structure of the neural network
#' @param theta List of parameters
#' @param normStat Normalized statistics
#' @param states List of states
#' @param x Input data
#' @param Sx Variance of input data
#' @param y Response data
#' @param dlayer Layer from which derivatives will be in respect to
#' @return - List of parameters
#' @return - List of normalized statistics
#' @return - Mean of predicted responses
#' @return - Variance of predicted responses
#' @return - Mean of first derivative of predicted responses
#' @return - Variance of first derivative of predicted responses
#' @return - Covariance between derivatives and inputs
#' @return - Mean of second derivative of predicted responses
#' @export
batchDerivative <- function(NN, theta, normStat, states, x, Sx, y, dlayer){
  # Initialization
  numObs = nrow(x)
  numDataPerBatch = NN$repBatchSize * NN$batchSize
  numCovariates = NN$nx
  mzl = matrix(0, numObs, NN$ny)
  Szl = matrix(0, numObs, NN$ny)
  mdg = matrix(0, numObs, numCovariates)
  Sdg = mdg
  Cdgz = mdg
  mddg = mdg
  # Loop
  loop = 0

  for (i in seq(from = 1, to = numObs, by = numDataPerBatch)){
    loop = loop + 1
    if (numDataPerBatch == 1){
      idxBatch = i:(i + NN$batchSize - 1)
    } else {
      if ((numObs - i) >= numDataPerBatch){
        idxBatch = i:(i + numDataPerBatch - 1)
      } else{
        idxBatch = c(i:numObs, sample(i - 1, numDataPerBatch - numObs + i -1))
      }
    }
    # Covariates
    xloop = matrix(t(x[idxBatch,]), nrow = NN$batchSize * numCovariates, ncol = NN$repBatchSize)
    Sxloop = matrix(t(Sx[idxBatch]), nrow = NN$batchSize * numCovariates, ncol = NN$repBatchSize)
    states = initializeInputs(states, xloop, Sxloop, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NN$xsc)

    # Training
    if (NN$trainMode == 1){
      yloop = matrix(t(y[idxBatch,]), nrow = NN$batchSize * NN$ny, ncol = NN$repBatchSize)
      out_feedForwardPass = feedForwardPass(NN, theta, states)
      states = out_feedForwardPass[[1]]
      mda = out_feedForwardPass[[2]]
      Sda = out_feedForwardPass[[3]]
      mdda = out_feedForwardPass[[4]]
      Sdda = out_feedForwardPass[[5]]
      # out_derivative = derivative(NN, theta, states, mda, Sda, mdda, Sdda, dlayer)
      # mdgi = out_derivative[[1]]
      # Sdgi = out_derivative[[2]]
      # Cdgzi = out_derivative[[3]]
      # mddgi = out_derivative[[4]]
      out_hiddenStateBackwardPass = hiddenStateBackwardPass(NN, theta, states, yloop, NULL, NULL)
      deltaM = out_hiddenStateBackwardPass[[1]]
      deltaS = out_hiddenStateBackwardPass[[2]]
      dtheta = parameterBackwardPass(NN, theta, states, deltaM, deltaS)
      theta = globalParameterUpdate(theta, dtheta)
    }
    # Testing
    else {
      out_feedForwardPass = feedForwardPass(NN, theta, states)
      states = out_feedForwardPass[[1]]
      mda = out_feedForwardPass[[2]]
      Sda = out_feedForwardPass[[3]]
      mdda = out_feedForwardPass[[4]]
      Sdda = out_feedForwardPass[[5]]
      out_derivative = derivative(NN, theta, states, mda, Sda, mdda, Sdda, dlayer)
      mdgi = out_derivative[[1]]
      Sdgi = out_derivative[[2]]
      Cdgzi = out_derivative[[3]]
      mddgi = out_derivative[[4]]

      out_extractStates = extractStates(states)
      ma = out_extractStates[[3]]
      Sa = out_extractStates[[4]]
      mzl[idxBatch,] = t(matrix(ma[[length(ma),1]], NN$ny, numDataPerBatch))
      Szl[idxBatch,] = t(matrix(Sa[[length(Sa),1]], NN$ny, numDataPerBatch))
      mdg[idxBatch,] = t(matrix(mdgi[[dlayer,1]], NN$nodes[dlayer], numDataPerBatch))
      Sdg[idxBatch,] = t(matrix(Sdgi[[dlayer,1]], NN$nodes[dlayer], numDataPerBatch))
      Cdgz[idxBatch,] = t(matrix(Cdgzi[[dlayer,1]], NN$nodes[dlayer], numDataPerBatch))
      mddg[idxBatch,] = t(matrix(mddgi[[dlayer,1]], NN$nodes[dlayer], numDataPerBatch))
    }


  }
  outputs <- list(theta, normStat, mzl, Szl, mdg, Sdg, Cdgz, mddg)
  return(outputs)
}

#' Network initialization
#'
#' Verify and add components to the neural network structure.
#'
#' @param NN Lists the structure of the neural network
#' @return - NN with all required components
#' @return - States of all required elements to perform TAGI
#' @export
initialization <- function(NN){
  # Build indices
  NN <- initialization_net(NN)
  NN <- layerEncoder(NN)
  NN <- parameters(NN)
  # States
  states <- initializeStates(NN$nodes, NN$batchSize, NN$repBatchSize, NN$xsc)
  outputs <- list(NN, states)
  return(outputs)
}


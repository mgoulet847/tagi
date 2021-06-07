#' One iteration of the Tractable Approximate Gaussian Inference (TAGI) with Derivative Calculations
#'
#' This function goes through one learning iteration of the neural network model
#' using TAGI.
#'
#' @param NN List that contains the structure of the neural network
#' @param theta List of parameters
#' @param normStat TBD
#' @param states List of states
#' @param x Set of input data
#' @param Sx Variance of x
#' @param y Set of corresponding responses
#' @param dlayer Layer from which derivatives will be in respect to
#' @return A list that contains:
#' @return - \code{theta}: List of updated parameters
#' @return - \code{normStat}: updated TBD
#' @return - \code{mzl}: Predicted responses
#' @return - \code{Szl}: Variance vector of predicted responses
#' @return - \code{mdg}: Predicted derivatives
#' @return - \code{Sdg}: Variance vector of derivatives
#' @return - \code{Cdgz}: Covariance between derivatives and inputs
#' @return - \code{mddg}: Predicted second derivatives
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
      if (NN$collectDev > 1){
        NN$actDerivative = 1
        states_derivative = out_derivative[[5]]
        mda_derivative = out_derivative[[6]]
        Sda_derivative = out_derivative[[7]]
        out_derivative = derivative(NN, theta, states_derivative, mda_derivative, Sda_derivative, mdda, Sdda, dlayer)
        mddgi = out_derivative[[1]]
        NN$actDerivative = 0
      }
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
#' @param NN List that contains the structure of the neural network
#' @return NN NN with all required components
#' @return states States of all required elements to perform TAGI
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


# Data processing

#' Normalize data
#'
#' This function normalizes data before entering the neural network.
#'
#' @param xtrain Training set of input variables
#' @param ytrain Training set of responses
#' @param xtest Testing set of input variables
#' @param ytest Testing set of responses
#' @return - Normalized training set of input variables
#' @return - Normalized training set of responses
#' @return - Normalized testing set of input variables
#' @return - Normalized testing set of responses
#' @return - Mean vector of input variables from training set
#' @return - Covariance matrix of input variables from training set
#' @return - Mean vector of responses from training set
#' @return - Covariance matrix of responses from training set
#' @export
normalize <- function(xtrain, ytrain, xtest, ytest){
  mxtrain = colMeans(xtrain, na.rm=TRUE)
  sxtrain = apply(xtrain, 2, sd, na.rm=TRUE)
  idx = rep(0, length(sxtrain))
  idx[sxtrain == 0] = 1
  sxtrain[idx] = 1
  mytrain = colMeans(ytrain, na.rm=TRUE)
  sytrain = apply(ytrain, 2, sd, na.rm=TRUE)
  xntrain = (xtrain - matlab::repmat(mxtrain,nrow(xtrain),1))/matlab::repmat(sxtrain,nrow(xtrain),1)
  yntrain = (ytrain - matlab::repmat(mytrain,nrow(ytrain),1))/matlab::repmat(sytrain,nrow(ytrain),1)
  xntest = (xtest - matlab::repmat(mxtrain,nrow(xtest),1))/matlab::repmat(sxtrain,nrow(xtest),1)
  yntest = ytest

  outputs <- list(xntrain, yntrain, xntest, yntest, mxtrain, sxtrain, mytrain, sytrain)
  return(outputs)
}

#' Split data
#'
#' This function splits data into training and test sets.
#'
#' @param x Input data
#' @param y Response data
#' @param ratio Training ratio
#' @return - Training set of input variables
#' @return - Training set of responses
#' @return - Testing set of input variables
#' @return - Testing set of responses
#' @export
split <- function(x, y, ratio){
  numObs = nrow(x)
  idxobs = sample(numObs)
  idxTrainEnd = round(ratio * numObs)
  idxTrain = idxobs[1:idxTrainEnd]
  idxTest = idxobs[(idxTrainEnd+1):numObs]
  xtrain = matrix(x[idxTrain,], nrow = idxTrainEnd)
  ytrain = matrix(y[idxTrain,], nrow = idxTrainEnd)
  xtest = matrix(x[idxTest,], nrow = (numObs-idxTrainEnd))
  ytest = matrix(y[idxTest,], nrow = (numObs-idxTrainEnd))

  outputs <- list(xtrain, ytrain, xtest, ytest)
  return(outputs)
}

#' Denormalize data
#'
#' This function denormalizes response data processed by the neural network.
#'
#' @param yn Predicted responses
#' @param syn Variance of the predicted responses
#' @param myntrain Mean vector of responses from training set
#' @param syntrain Variance vector of responses from training set
#' @return - Mean of denormalized predicted responses
#' @return - Variance of denormalized predicted responses
#' @export
denormalize <- function(yn, syn, myntrain, syntrain){
  y = yn * syntrain + myntrain
  if (!(is.null(syn))){
    sy = syntrain^2 * syn
  } else {
    sy = NULL
  }
  outputs <- list(y, sy)
  return(outputs)
}


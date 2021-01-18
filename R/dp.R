# Data processing

#' Normalize data
#'
#' This function normalizes data before entering the neural network.
#'
#' @param xtrain Training set of input variables
#' @param ytrain Training set of responses
#' @param xtest Testing set of input variables
#' @param ytest Testing set of responses
#' @return A list that contains:
#' @return - The normalized training set of input variables
#' @return - The normalized training set of responses
#' @return - The normalized testing set of input variables
#' @return - The normalized testing set of responses
#' @return - Mean vector of input variables from training set
#' @return - Variance vector of input variables from training set
#' @return - Mean vector of responses from training set
#' @return - Variance vector of responses from training set
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
#' This function split data into training and test sets.
#'
#' @param x Input data
#' @param y Response data
#' @param ratio Training ratio
#' @return A list that contains:
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
  xtrain = x[idxTrain,]
  ytrain = matrix(y[idxTrain,], nrow = length(y[idxTrain,]))
  xtest = x[idxTest,]
  ytest = matrix(y[idxTest,], nrow = length(y[idxTest,]))

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
#' @return A list that contains:
#' @return - The denormalized predicted responses
#' @return - The variance of denormalized predicted responses
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


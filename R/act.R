#' Assign ID to Activation Functions
#'
#' This function assigns a number attached to the type of activation function.
#'
#' @param funName Type of activation function: "tanh", "sigm", "cdf", "relu" or
#' "softplus"
#' @return An ID number which corresponds to:
#' @return - 1 if \code{funName} is "tanh"
#' @return - 2 if \code{funName} is "sigm"
#' @return - 3 if \code{funName} is "cdf"
#' @return - 4 if \code{funName} is "relu"
#' @return - 5 if \code{funName} is "softplus"
#' @export
activationFunIndex <- function(funName){
  if (funName == "tanh"){
    funIdx = 1
  } else if (funName == "sigm"){
    funIdx = 2
  } else if (funName == "cdf"){
    funIdx = 3
  } else if (funName == "relu"){
    funIdx = 4
  } else if (funName == "softplus"){
    funIdx = 5
  }
  return(funIdx)
}

#' Calculate Mean of Activated Units
#'
#' This function uses lineratization to estimate the activation units mean vector
#' \eqn{\mu_{A}} and the Jacobian matrix evaluated at \eqn{\mu_{Z}}.
#'
#' @param z Vector of units for the current layer
#' @param mz Mean vector of the units for the current layer \eqn{\mu_{Z}}
#' @param funIdx Activation function index defined by \code{\link{activationFunIndex}}
#' @return A list that contains the activation units mean vector \eqn{\mu_{A}} and the Jacobian
#' matrix evaluated at \eqn{\mu_{Z}}
#' @export
meanA <- function(z, mz, funIdx){
  if (funIdx == 1){ # tanh
    dtanhf <- function(x){1 - tanh(x)^2}
    s = dtanhf(mz) * (z - mz) + tanh(mz)
    J = dtanhf(z)
  } else if (funIdx == 2){ # sigmoid
    sigmoid <- function(x){1 / (1 + exp(-x))}
    dsigmoid <- function(x){sigmoid(x) * (1 -sigmoid(x))}
    s = sigmoid(2)
    J= dsigmoid(2)
  } else if (funIdx == 3){ # cdf
    s = stats::dnorm(mz) * (z - mz) + stats::pnorm(mz)
    J = stats::dnorm(z)
  } else if (funIdx == 4){ # relu
    s = pmax(mz,0)
    J = matrix(0, nrow = nrow(z), ncol = ncol(z))
    J[z > 0] = 1
  } else if (funIdx == 5){ # softplus
    alpha = 10
    k = matrix(0, nrow = nrow(mz), ncol = ncol(mz))
    k[alpha * mz < 30] = 1
    s = 1 + exp (alpha * mz * k)
    s = (log(s) + mz * (1-k)) / alpha
    J = k * exp(alpha * mz * k) / (1 + exp(alpha * mz * k)) + (1 - k) / alpha
  }
  outputs <- list(s, J)
  return(outputs)
}

#' Calculate Variance of Activated Units
#'
#' This function uses lineratization to estimate the covariance matrix of activation units \eqn{\Sigma_{A}}.
#'
#' @param J Jacobian matrix evaluated at \eqn{\mu_{Z}}
#' @param Sz Covariance matrix of the units for the current layer \eqn{\Sigma_{Z}}
#' @return The activation units variance vector \eqn{\Sigma_{A}}
#' @export
covarianceSa <- function(J, Sz){
  Sa = J * Sz * J
  return(Sa)
}

#' Mean, Jacobian and Variance of Activated Units
#'
#' This function returns mean vector \eqn{\mu_{A}}, Jacobian matrix evaluated at
#' \eqn{\mu_{Z}} and covariance matrix of activation units \eqn{\Sigma_{A}}.
#'
#' @param z Vector of units for the current layer
#' @param mz Mean vector of the units for the current layer \eqn{\mu_{Z}}
#' @param Sz Covariance matrix of the units for the current layer \eqn{\Sigma_{Z}}
#' @param funIdx Activation function index defined by \code{\link{activationFunIndex}}
#' @return The activation units mean vector \eqn{\mu_{A}}
#' @return The activation units variance vector \eqn{\Sigma_{A}}
#' @return Jacobian matrix evaluated at \eqn{\mu_{Z}}
#' @export
meanVar <- function(z, mz, Sz, funIdx){
  out_meanA <- meanA(z, mz, funIdx)
  m = out_meanA[[1]]
  J = out_meanA[[2]]
  out_covarianceSa <- covarianceSa(J, Sz)
  S = out_covarianceSa[[1]]
  outputs <- list(m, S, J)
  return(outputs)
}

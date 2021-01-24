# Metric for performance evaluation

#' Compute error
#'
#' This function calculates the Root Mean Square Error (RMSE). It takes as input
#' two vectors (or matrices) with one containing the real \eqn{y}'s and the other the
#' predicted \eqn{y}'s from the model.
#'
#' @param y A vector or a matrix of the real \eqn{y}'s
#' @param ypred A vector or a matrix of the predicted \eqn{y}'s from the model
#' @return RMSE for the given data
#' @export
computeError <- function(y, ypred){
  e = sqrt(mean((y-ypred)^2))
  return(e)
}

#' Compute log-likelihood
#'
#' This function calculates the log-likelihood (LL). It takes as input
#' three vectors (or matrices) with one containing the real \eqn{y}'s, one with the
#' predicted \eqn{y}'s from the model and the last one with the variance of the \eqn{y}'s.
#'
#' @param y A vector or a matrix of the real \eqn{y}'s
#' @param ypred A vector or a matrix of the predicted \eqn{y}'s form the model
#' @param Vpred A vector or a matrix of the variance of the predicted \eqn{y}'s form
#' the model
#' @return LL for the given data
#' @export
loglik <- function(y, ypred, Vpred){
  d = ncol(y)
  if (d==1){
    LL = mean(-0.5*log(2*pi*Vpred) - (0.5*(y-ypred)^2)/Vpred)
  }
  return(LL)
}

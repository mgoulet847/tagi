% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mt.R
\name{loglik}
\alias{loglik}
\title{Compute log-likelihood}
\usage{
loglik(y, ypred, Vpred)
}
\arguments{
\item{y}{A vector or a matrix of the real \eqn{y}'s}

\item{ypred}{A vector or a matrix of the predicted \eqn{y}'s form the model}

\item{Vpred}{A vector or a matrix of the variance of the predicted \eqn{y}'s form
the model}
}
\value{
LL for the given data
}
\description{
This function calculates the log-likelihood (LL). It takes as input
three vectors (or matrices) with one containing the real \eqn{y}'s, one with the
predicted \eqn{y}'s from the model and the last one with the variance of the \eqn{y}'s.
}

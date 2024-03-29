% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tagi.R
\name{parameterBackwardPass}
\alias{parameterBackwardPass}
\title{Backpropagation (parameters' deltas)}
\usage{
parameterBackwardPass(NN, theta, states, deltaM, deltaS)
}
\arguments{
\item{NN}{Lists the structure of the neural network}

\item{theta}{List of parameters}

\item{states}{List of states}

\item{deltaM}{Delta of mean vector of units given \eqn{y} \eqn{\mu_{Z}|y} at all layers}

\item{deltaS}{Delta of covariance matrix of units given \eqn{y} \eqn{\Sigma_{Z}|y} at all layers}
}
\value{
Parameters' deltas (mean and covariance for each)
}
\description{
This function calculates parameter's deltas.
}

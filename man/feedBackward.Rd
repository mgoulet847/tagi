% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tagi.R
\name{feedBackward}
\alias{feedBackward}
\title{Backpropagation}
\usage{
feedBackward(NN, mp, Sp, mz, Sz, Czw, Czb, Czz, y)
}
\arguments{
\item{NN}{List that contains the structure of the neural network}

\item{mp}{List that contains mean vectors of the parameters for each layer \eqn{\mu_{\theta}}}

\item{Sp}{List that contains covariance matrices of the
parameters for each layer \eqn{\Sigma_{\theta}}}

\item{mz}{List that contains the mean vectors of the units
for each layer \eqn{\mu_{Z}}}

\item{Sz}{List that contains the covariance matrices of the
units for each layer \eqn{\Sigma_{Z}}}

\item{Czw}{List that contains the covariance matrices between
units and weights for each layer \eqn{\Sigma_{ZW}}}

\item{Czb}{List that contains the covariance matrices between
units and biases for each layer \eqn{\Sigma_{ZB}}}

\item{Czz}{List that contains the covariance matrices between units of the
previous and current layers for each layer \eqn{\Sigma_{ZZ^{+}}}}

\item{y}{A vector or a matrix of responses}
}
\value{
A list that contains:

\itemize{
\item List that contains the updated mean vectors of the parameters
for each layer
}

\itemize{
\item List that contains the updated covariance matrices of the
parameters for each layer
}
}
\description{
This function feeds the neural network backward from responses to input data.
}
\seealso{
\code{\link{backwardHiddenStateUpdate}},
\code{\link{backwardParameterUpdate}}, \code{\link{forwardHiddenStateUpdate}}
}

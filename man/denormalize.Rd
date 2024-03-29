% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dp.R
\name{denormalize}
\alias{denormalize}
\title{Denormalize data}
\usage{
denormalize(yn, syn, myntrain, syntrain)
}
\arguments{
\item{yn}{Predicted responses}

\item{syn}{Variance of the predicted responses}

\item{myntrain}{Mean vector of responses from training set}

\item{syntrain}{Variance vector of responses from training set}
}
\value{
\itemize{
\item Mean of denormalized predicted responses
}

\itemize{
\item Variance of denormalized predicted responses
}
}
\description{
This function denormalizes response data processed by the neural network.
}

% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tagi.R
\name{fcCovaz}
\alias{fcCovaz}
\title{Covariance between activation and hidden units}
\usage{
fcCovaz(Jo, J, Sz, mw, ni, no, B)
}
\arguments{
\item{Jo}{Jacobian of next layer}

\item{J}{Jacobian of current layer}

\item{Sz}{Covariance of units from current layer}

\item{mw}{Mean vector of weights for the current layer}

\item{ni}{Number of units in current layer}

\item{no}{Number of units in next layer}

\item{B}{Batch size}
}
\value{
\itemize{
\item Covariance between activation and hidden layers (same layer)
}

\itemize{
\item Covariance between activation (next) and hidden (current) layers
}
}
\description{
This function calculates covariance between activation and hidden units.
}

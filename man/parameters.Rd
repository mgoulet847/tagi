% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/indices.R
\name{parameters}
\alias{parameters}
\title{Indices for biases and weights}
\usage{
parameters(NN)
}
\arguments{
\item{NN}{List that contains the structure of the neural network}
}
\value{
NN with three new elements, each of size (number of layers -1) :

\itemize{
\item Weight indices for each layer
}

\itemize{
\item Bias indices for each layer
}

\itemize{
\item Combined weight and bias indices for each layer
}
}
\description{
This function assigns indices for all weights and biases in
the neural network.
}
\details{
Bias indices are assigned from 1 to the maximum number of biases
for a given layer. Then, weight indices start where bias indices end plus one until
all weights are assigned an indice. The number of weights for a given layer is
the number of units in the previous layer times the number of units in the
current one.

For example, if there are 10 units in the previous layer and 50
in the current one, then there would be 50 biases and 500 weights in the
current layer. The bias indices would be from 1 to 50 and weight IDs from 51 to
550.
}

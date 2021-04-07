#' Network initialization
#'
#' Verify and add components to the neural network structure.
#'
#' @param NN List that contains the structure of the neural network
#' @return NN NN with all required components
#' @return states States of all required elements to perform TAGI
#' @export
initialization <- function(NN){
  # Build indices
  NN <- initialization_net(NN)
  NN <- layerEncoder(NN)
  NN <- parameters(NN)
  # States
  states <- initializeStates(NN$nodes, NN$batchSize, NN$repBatchSize, NN$xsc)
  outputs <- list(NN, states)
  return(outputs)
}


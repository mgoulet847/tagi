#' Network initialization
#'
#' Verify and add components to the neural network structure.
#'
#' @param NN List that contains the structure of the neural network
#' @return NN with all required components
#' @export
initialization_net <- function(NN){
  if (!any(NN$layer %in% NN)){
    NN$layer = rep(1, length(NN$nodes))
  }
  numLayers = length(NN$layer)
  if (!any(NN$task %in% NN)){
    NN$task = "regression"
  }
  if (NN$task %in% c("classification", "regression")){
    if (!any(NN$filter %in% NN)){
      NN$filter = rep(0, numLayers)
    }
    if (!any(NN$kernelSize %in% NN)){
      NN$kernelSize = rep(0, numLayers)
    }
    if (!any(NN$padding %in% NN)){
      NN$padding = rep(0, numLayers)
    }
    if (!any(NN$paddingType %in% NN)){
      NN$paddingType = rep(0, numLayers)
    }
    if (!any(NN$stride %in% NN)){
      NN$stride = rep(0, numLayers)
    }
    if (!any(NN$actFunIdx %in% NN)){
      NN$actFunIdx = rep(0, numLayers)
    }
    if (!any(NN$xsc %in% NN)){
      NN$xsc = rep(0, numLayers)
    }
    if (!any(NN$imgH %in% NN)){
      NN$imgH = rep(0, numLayers)
      if (any(NN$imgSize %in% NN)){
        NN$imgH[1] = NN$imgSize[1]
      }
    }
    if (!any(NN$obsShow %in% NN)){
      NN$obsShow = 10000
    }
    if (!any(NN$epsilon %in% NN)){
      NN$epsilon = 0.0001
    }
    if (!any(NN$normMomentum %in% NN)){
      NN$normMomentum = 0.9
    }
    if (!any(NN$da %in% NN)){
      da <- list(enable = 0, p = 0.5, types = NULL)
      NN$da <- da
    }
    if (!any(NN$svDecayFactor %in% NN)){
      NN$svDecayFactor = 0.8
    }
    if (!any(NN$svmin %in% NN)){
      NN$svmin = 0
    }
    if (!any(NN$earlyStop %in% NN)){
      NN$earlyStop = 0
    }
    if (!any(NN$displayMode %in% NN)){
      NN$displayMode = 1
    }
    if (!any(NN$errorRateEval %in% NN)){
      NN$errorRateEval = 1
    }
    if (!any(NN$numDevices %in% NN)){
      NN$numDevices = 1
    }
  }
  if (!any(NN$collectDev %in% NN)){
    NN$collectDev = 0
  }
  if (!any(NN$initParamType %in% NN)){
    NN$initParamType = "Xavier"
  }
  if (!any(NN$gainM %in% NN)){
    NN$gainM = rep(1, numLayers - 1)
  }
  if (!any(NN$gainS %in% NN)){
    if (NN$initParamType == "Xavier"){
      NN$gainS = rep(1, numLayers - 1)
    }
    else if (NN$initParamType == "He"){
      NN$gainS = rep(2, numLayers - 1)
    }
  }
  if (!any(NN$maxEpoch %in% NN)){
    NN$maxEpoch = 10
  }
  if (!any(NN$convariateEstm %in% NN)){
    NN$convariateEstm = 0
  }
  if (!any(NN$learnSv %in% NN)){
    NN$learnSv = 0
  }
  if (!any(NN$imgSize %in% NN)){
    NN$imgSize = rep(0, numLayers - 1)
  }
  if (!any(NN$savedEpoch %in% NN)){
    NN$savedEpoch = 0
  }
  if (!any(NN$learningRateSchedule %in% NN)){
    NN$learningRateSchedule = 0
  }
  if (!any(NN$scheduleSv %in% NN)){
    NN$scheduleSv = 0
  }
  if (!any(NN$lastLayerUpdate %in% NN)){
    NN$lastLayerUpdate = 1
  }
  NN <- NN
}

#' Layer Encoder
#'
#' Add layer encoder to the neural network structure.
#'
#' @param NN List that contains the structure of the neural network
#' @return NN with layer encoder
#' @export
layerEncoder <- function(NN){
  layerEncoder <- list("fc" = 1)
  NN$layerEncoder <- layerEncoder
  NN <- NN
}

#' Indices for biases and weights
#'
#' This function assigns indices for all weights and biases in
#' the neural network.
#'
#' @details Bias indices are assigned from 1 to the maximum number of biases
#' for a given layer. Then, weight indices start where bias indices end plus one until
#' all weights are assigned an indice. The number of weights for a given layer is
#' the number of units in the previous layer times the number of units in the
#' current one.
#' @details For example, if there are 10 units in the previous layer and 50
#' in the current one, then there would be 50 biases and 500 weights in the
#' current layer. The bias indices would be from 1 to 50 and weight IDs from 51 to
#' 550.
#'
#' @param NN List that contains the structure of the neural network
#' @return NN with three new elements, each of size (number of layers -1) :
#' @return - \code{idxb}: List of the weight indices for each layer
#' @return - \code{idxw}: List of the bias indices for each layer
#' @return - \code{idxwb}: List of the combined weight and bias indices for each layer
#' @export
parameters <- function(NN){
  # Initialization
  nodes = NN$nodes
  layer = NN$layer
  numLayers = length(nodes)
  # Bias
  idxb = matrix(list(), nrow = numLayers - 1, ncol = 1)
  # Weights
  idxw = matrix(list(), nrow = numLayers - 1, ncol = 1)
  idxwXsc = matrix(list(), nrow = numLayers - 1, ncol = 1)
  idxbXsc = matrix(list(), nrow = numLayers - 1, ncol = 1)
  # Bias and Weights
  idxbw = matrix(list(), nrow = numLayers - 1, ncol = 1)
  # Total number of parameters
  numParams = matrix(list(), nrow = numLayers - 1, ncol = 1)
  numParamsPerLayer = matrix(1, numLayers, numLayers - 1)
  totalNumParams = 0

  for (j in 1:(numLayers - 1)){
    numParams[[j, 1]] = nodes[j+1] + nodes[j+1] * nodes[j]
    if ((NN$collectDev >= 1) & (layer[j+1] == 1)){
      idxb[[j, 1]] = matrix(1:(nodes[j+1]), nrow = nodes[j+1], ncol = 1)
      idxw[[j, 1]] = matrix(1:(nodes[j+1] * nodes[j]), nrow = nodes[j+1] * nodes[j], ncol = 1)
    } else {
      idxbw[[j, 1]] = matrix(1:numParams[[j, 1]], 1, numParams[[j, 1]])
      idxb[[j, 1]] = matrix(idxbw[[j, 1]][1, 1:(nodes[j+1])], nrow = nodes[j+1], ncol = 1)
      idxw[[j, 1]] = matrix(idxbw[[j, 1]][1, ((nodes[j+1]) + 1):numParams[[j, 1]]], nrow = (numParams[[j, 1]] - nodes[j+1]), ncol = 1)
    }
    if (!(is.null(idxw[[j, 1]]))){
      numParamsPerLayer[1,j] = nrow(idxw[[j, 1]])
    }

    if (!(is.null(idxb[[j, 1]]))){
      numParamsPerLayer[2,j] = nrow(idxb[[j, 1]])
    }

    if (!(is.null(numParams[[j, 1]]))){
      totalNumParams = totalNumParams + numParams[[j, 1]]
    }
  }

  numParamsPerLayer_2 = cbind(matrix(0, nrow(numParamsPerLayer), 1), numParamsPerLayer)
  numParamsPerLayer_2 = t(apply(numParamsPerLayer_2,1,cumsum))

  NN$idxb = NULL
  NN$idxw = NULL
  NN$idxbw = NULL
  NN$idxwXsc = NULL
  NN$idxbXsc = NULL
  NN$totalNumParams = NULL
  NN$numParamsPerLayer = NULL
  NN$numParamsPerLayer_2 = NULL
  NN <- c(NN,idxb = list(idxb), idxw = list(idxw), idxbw = list(idxbw),
          idxwXsc = list(idxwXsc), idxbXsc = list(idxbXsc),
          totalNumParams = list(totalNumParams),
          numParamsPerLayer = list(numParamsPerLayer),
          numParamsPerLayer_2 = list(numParamsPerLayer_2))
}

#' Indices for covariances in the neural network
#'
#' This function assigns indices for all covariance elements in
#' the neural network.
#'
#' @param NN List that contains the structure of the neural network
#' @return NN with new elements:
#' @return - \code{idxFmwa}: Lists of indices (weights and activation units) for
#' deterministic matrix F *  \eqn{\mu_{WA}} for each layer
#' @return - \code{idxFmwab}: List of bias indices for deterministic matrix F *  \eqn{\mu_{B}} for
#' each layer
#' @return - \code{idxFCzwa}: Lists of indices (weights and activation units) for
#' deterministic matrix F *  \eqn{\Sigma_{ZWA}} for each layer
#' @return - \code{idxSzpUd}: List of indices for the parameter update step for each layer
#' @return - \code{idxSzzUd}: List of indices for the hidden state update step for each
#' layer
#' @return - \code{idxFCwwa}: Lists of indices (weights and activation units) for deterministic matrix F *  \eqn{\Sigma_{WA\theta}} for
#' each layer
#' @return - \code{idxa}: List of indices for activation unit for each layer
#' @return - \code{idxFCb}: List of bias indices for deterministic matrix F *  \eqn{\Sigma_{B}} for
#' each layer
#' @export
covariance <- function(NN){
  # Initialization
  batchSize = NN$batchSize
  nodes = NN$nodes
  numLayers = length(nodes)
  # Indices for F*mwa
  idxFmwa = matrix(list(), nrow = numLayers - 1, ncol = 2)
  idxFmwab = matrix(list(), nrow = numLayers - 1, ncol = 1)
  # Indices for F*Czwa
  idxFCzwa = matrix(list(), nrow = numLayers - 1, ncol = 2)
  # Indices for activation unit
  idxa = matrix(list(), nrow = numLayers - 1, ncol = 1)
  # Indices for F*Cwwa
  idxFCwwa = matrix(list(), nrow = numLayers - 1, ncol = 2)
  # Indices for F*Cb
  idxFCb = matrix(list(), nrow = numLayers - 1, ncol = 2)

  # Indices for updating parameters between layers
  idxSzpUd = matrix(list(), nrow = numLayers - 1, ncol = 1)
  # Indices for updating hidden states between layers
  idxSzzUd = matrix(list(), nrow = numLayers, ncol = 1)

  for (j in 1:(numLayers - 1)){

    # Loop initialization
    dnext = batchSize * nodes[j+1]
    idxa[[j, 1]] = 1:nodes[j] * batchSize
    idxa[[j, 1]] = t(t(idxa[[j, 1]]))
    idxaNext = matrix(1:nodes[j+1] * batchSize, 1, nodes[j+1])
    mode(idxa[[j, 1]]) <- "integer"
    mode(idxaNext) <- "integer"

    # Get indices for F*mwa
    idxFmwa_1 = matlab::repmat(matrix(t(matrix(NN$idxw[[j, 1]], nodes[j+1], nodes[j])), nrow = 1, ncol = nodes[j+1] * nodes[j]), 1, batchSize)
    idxFmwa_2 = matrix(matlab::repmat(matrix(idxa[[j, 1]], nodes[j], batchSize), nodes[j+1])[,1], nrow = 1, ncol = dnext * nodes[j])

    idxFmwa[[j, 1]] = t(matrix(t(idxFmwa_1), nodes[j], dnext))
    idxFmwa[[j, 2]] = t(matrix(t(idxFmwa_2), nodes[j], dnext))

    # Get indices for F*b
    idxFmwab[[j, 1]] = matlab::repmat(NN$idxb[[j, 1]], batchSize, 1)

    # Get indices for F*Czwa
    if (!(is.null(NN$sx)) || j > 1){
      idxFCzwa[[j, 1]] = matlab::repmat(NN$idxw[[j, 1]], batchSize, 1)
      idxFCzwa[[j, 2]] = matrix(matlab::repmat(t(idxa[[j, 1]]), nodes[j+1], 1), nrow = length(idxa[[j, 1]]) * nodes[j+1], ncol = 1)
      mode(idxFCzwa[[j, 1]]) <- "integer"
      mode(idxFCzwa[[j, 2]]) <- "integer"
    }

    # Get indices for the parameter update step
    idxSzpUd[[j, 1]] = do.call(rbind, replicate(nodes[j] + 1, matrix(idxaNext, nodes[j+1], batchSize), simplify=FALSE))

    # Get indices for F*Cwwa
    # Indices for Sp that uses to evaluate Cwwa
    idxFCwwa[[j, 1]] = matrix(matlab::repmat(t(NN$idxw[[j, 1]]), batchSize, 1), nrow = nodes[j] * dnext, ncol = 1)
    # Indices for ma that uses to evaluate Cwwa
    idxFCwwa[[j, 2]] = matrix(matlab::repmat(matrix(idxa[[j, 1]], nodes[j], batchSize), nodes[j+1], 1), nrow = nodes[j] * dnext, ncol = 1)

    # Get indices for F*Sb
    idxFCb[[j, 1]] = matrix(matlab::repmat(t(NN$idxb[[j, 1]]), batchSize, 1), nrow = dnext, ncol = 1)

    # Get indices for the hidden state update step
    idxSzzUd[[j, 1]] = t(matrix(matlab::repmat(t(matrix(idxaNext, nodes[j+1], batchSize)), nodes[j], 1), nrow = nodes[j+1], ncol = nodes[j] * batchSize))

    # Integer matrix (takes less space)
    mode(idxFmwa[[j, 1]]) <- "integer"
    mode(idxFmwa[[j, 2]]) <- "integer"
    mode(idxFmwab[[j, 1]]) <- "integer"
    mode(idxSzpUd[[j, 1]]) <- "integer"
    mode(idxFCwwa[[j, 1]]) <- "integer"
    mode(idxFCwwa[[j, 2]]) <- "integer"
    mode(idxFCb[[j, 1]]) <- "integer"
    mode(idxSzzUd[[j, 1]]) <- "integer"
  }

  # Outputs
  # F*mwa
  NN$idxFmwa = NULL
  NN$idxFmwab = NULL
  # F*Cawa
  NN$idxFCzwa = NULL
  NN$idxSzpUd = NULL
  # Caa
  NN$idxSzzUd = NULL
  # F*Cwwa
  NN$idxFCwwa = NULL
  # a
  NN$idxa = NULL
  NN$idxFCb = NULL

  NN <- c(NN, idxFmwa = list(idxFmwa),
          idxFmwab = list(idxFmwab),
          idxFCzwa = list(idxFCzwa),
          idxSzpUd = list(idxSzpUd),
          idxSzzUd = list(idxSzzUd),
          idxFCwwa = list(idxFCwwa),
          idxa = list(idxa),
          idxFCb = list(idxFCb)
  )
}

twoPlus <- function(m, S, deltaM, deltaS){
  m = m + deltaM
  S = S + deltaS
  outputs <- list(m, S)
  return(outputs)
}

vectorizedMeanVar <- function(ma, mp, Sa, Sp){
  Sz = Sp*ma*ma + Sa*Sp + Sa*mp*mp
  mz = ma*mp
  outputs <- list(mz, Sz)
  return(outputs)
}

vectorizedDelta <- function(C, deltaM, deltaS){
  deltaM = C*deltaM
  deltaS = C*deltaS*C
  outputs <- list(deltaM, deltaS)
  return(outputs)
}

vectorized4Delta <- function(W, C1, C2, deltaM, deltaS){
  deltaM1 = W*C1*deltaM
  deltaS1 = W*C1*deltaS*W*C1
  deltaM2 = W*C2*deltaM
  deltaS2 = W*C2*deltaS*W*C2

  outputs <- list(deltaM1, deltaS1, deltaM2, deltaS2)
  return(outputs)
}

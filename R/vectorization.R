vectorizedMeanVar <- function(ma, mp, Sa, Sp){
  Sz = Sp*ma*ma + Sa*Sp + Sa*mp*mp
  mz = ma*mp
  outputs <- list(mz, Sz)
  return(outputs)
}

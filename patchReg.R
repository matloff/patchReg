
# NEXT
# 
# add holdout, check whether this method works well
# 
# allow any regest ftn, e.g. lm, not just qe*
# 
# Y = 0,1 case

# github!!!!!; announcde

kmPatch <- function(XYdata,yName,numClust,holdout = floor(min(1000,
            0.1*nrow(XYdata)))) 
{

   yCol <- which(names(XYdata) == yName)
   xData <- XYdata[,-yCol]
   yData <- XYdata[,yCol]
   if (any(sapply(xData,class) != 'numeric'))
      xData <- regtools::factorsToDummies(xData,omitLast=TRUE,dfOut=TRUE)
   numXvars <- ncol(xData)
   newData <- cbind(xData,yData)
   names(newData)[numXvars+1] <- yName
   if (!is.null(holdout)) {
      tstIdxs <- sample(1:nrow(newData), holdout)
      tstNewData <- newData[tstIdxs,,drop=FALSE]
      newData <- newData[-tstIdxs,,drop=FALSE]
   }
   kmOut <- kmeans(newData[,1:numXvars],numClust)
   clustNums <- as.factor(kmOut$cluster)
   clustLevels <- levels(clustNums)
   kmPatchOut <- lapply(clustLevels,
         function(clustNum) qePolyLin(newData[clustNums==clustNum,],yName))
   class(kmPatchOut) <- 'kmpatchout'
   kmPatchOut$clustLevels <- clustLevels
   kmPatchOut$centers <- kmOut$centers
   preds <- predict(kmPatchOut,tstNewData[,1:numXvars])
   kmPatchOut$testAcc <- mean(abs(tstNewData[,numXvars+1]-preds))
   kmPatchOut
}

predict.kmpatchout <- function(object,newX) 
{
   if (any(sapply(newX,class) != 'numeric'))
      newX <- regtools::factorsToDummies(newX,omitLast=TRUE,dfOut=TRUE)
   npreds <- nrow(newX)
   preds <- vector(length=npreds)
   for (i in 1:npreds) {
      newx <- newX[i,]
      closestIndex <- FNN::get.knnx(object$centers,newx,k=1)$nn.index
      preds[i] <- predict(object[[closestIndex]],newx)
   }
   preds
}


# TODO:  (a) classification case, (b) only 1 X variable, (c) factors to
# numeric conversion of factors in new obs

# goal of this code: improve the predictive accuracy of a
# regression/classification procedure by breaking the X space into
# patches and doing separate fits on each patch; currently k-means is
# used to create the patches

# requirements

# 1. the fitting function must have a 'data' argument, as with 'lm' and 'ranger'
# 
# 2. the return value of the fitting function must have a 'predict' method
# 
# 3. no character or logical variables

patchReg <- function(XYdata,yName,numClust,regCall,
   savePreds=FALSE,regPredName=NULL,classPredName=NULL,
   holdout = floor(min(1000,0.1*nrow(XYdata)))) 
{

   yCol <- which(names(XYdata) == yName)
   Xdata <- XYdata[,-yCol]
   Ydata <- XYdata[,yCol]
   classif <- class(Ydata) == 'factor'
   
   # convert predictor-variable factors to dummies 
   if (any(sapply(Xdata,class) != 'numeric')) {
      Xdata <- regtools::factorsToDummies(Xdata,omitLast=TRUE,dfOut=TRUE)
      factorsInfo <- attr(Xdata,'factorsInfo')
   } else factorsInfo <- NULL
   numXvars <- ncol(Xdata)
   newData <- cbind(Xdata,Ydata)
   names(newData)[numXvars+1] <- yName

   if (!is.null(holdout)) {
      tstIdxs <- sample(1:nrow(newData), holdout)
      tstXData <- newData[tstIdxs,1:numXvars,drop=FALSE]
      tstYData <- newData[tstIdxs,numXvars+1,drop=FALSE]
      trnXData <- newData[-tstIdxs,1:numXvars,drop=FALSE]
      trnYData <- newData[-tstIdxs,numXvars+1,drop=FALSE]
      trnXYData <- newData[-tstIdxs,]
   } else {
      trnXData <- newData[,1:numXvars,drop=FALSE]
      tstYData <- newData[,numXvars+1,drop=FALSE]
   }

   # form clusters
   kmeansOut <- kmeans(trnXData,numClust)
   clustInfo <- findClusters(kmeansOut,trnXData)
   # clusters in terms of the origin data
   clustData <- lapply(clustInfo$clusterIndices,
      function(indices) trnXYData[indices,])

   # do separate fits to the clusters
   pROut <- lapply(clustData,evalr(regCall))
   # element i of pROut is the value returned by calling regCall on
   # cluster i
   pROut$centers <- kmeansOut$centers
   pROut$clustNums <- clustInfo$clustNums
   pROut$factorsInfo <- factorsInfo
   pROut$classif <- classif
   pROut$regPredName <- regPredName
   pROut$classPredName <- classPredName
   class(pROut) <- 'prout'

   if (holdout != 0) {
      preds <- predict(pROut,tstXData)
      if (!classif)
         pROut$testAcc <- mean(abs(tstYData[,1] - preds))
      else 
         pROut$testAcc <- mean(tstYData[,1] != preds)
      if(savePreds) {
         preds <- cbind(clustInfo$clustNums,preds)
         preds <- as.data.frame(preds)
         names(preds) <- c('clustNums','preds')
         pROut$preds <- preds
      }
   }

   pROut
}

## predict.prout <- function(object,predXData) 
## {
##    tmp <- FNN::get.knnx(query=newxData,data=object$centers,k=1)
##    indices <- tmp$nn.index
##    tmp <- cbind(newxData,indices)
##    applyFtn <- function(rowi) 
##    {
##       nc <- ncol(tmp)
##       j <- rowi[nc]
##       predict(object[[j]],rowi[-nc])
##    }
##    apply(predXData,1,applyFtn)
## }

plot.prout <- function() 
{

}

findClusters <- function(kmout,trnXdata) 
{
   clustInfo <- list()
   # for each row in trnXdata, find the closest cluster center index
   tmp <- FNN::get.knnx(query=trnXdata,data=kmout$centers,k=1)
   clustInfo$clustNums <- tmp$nn.index
   # now find the clusters themselves, in terms of index numbers in
   # Xdata
   clustInfo$clusterIndices <- split(1:nrow(trnXdata),clustInfo$clustNums)
   clustInfo
}

predict.prout <- function(object,newX) 
{
   if (any(sapply(newX,class) != 'numeric'))
      newX <- regtools::factorsToDummies(newX,omitLast=TRUE,dfOut=TRUE,
         factorsInfo=object$factorsInfo)
   npreds <- nrow(newX)
   preds <- vector(length=npreds)
   for (i in 1:npreds) {
      newx <- newX[i,]
      closestIdx <- FNN::get.knnx(object$centers,newx,k=1)$nn.index
      tmp <- predict(object[[closestIdx]],newx)
      if (object$classif) {
         # tmp <- tmp$predClasses  # assumes qeML
         classPredName <- object$classPredName
         if (!is.null(classPredName)) {
            tmp <- tmp[[classPredName]]
            if (substr(tmp,1,4) == 'dfr.')  # due to factorsToDummies
               tmp <- substr(tmp,5,nchar(tmp))
         }
      } else {  # regression case
         regPredName <- object$regPredName
         if (!is.null(regPredName)) tmp <- tmp[[regPredName]]
      }
      preds[i] <- tmp
   }
   preds
}

# examples
# patchReg(svcensus,"wageinc",4,"function(xy) lm(wageinc~.,data=xy)")
# patchReg(svcensus,"gender",2,"function(xy) qeLogit(xy,'gender')")
# patchReg(svcensus,"wageinc",8,"function(xy) qeKNN(xy,'wageinc',k=100)")
# patchReg(svcensus,"gender",2,"function(xy) qeKNN(xy,'gender',k=100,
#   classPredName='predClasses')")
# patchReg(svcensus,"occ",2,"function(xy) qeKNN(xy,'occ',k=100,
#   classPredName='predClasses')")
# patchReg(svcensus,"gender",2,"function(xy) qeLASSO(xy,'gender')",
#   classPredName='predClasses')
# patchReg(svcensus,"wageinc",2,"function(xy) ranger(wageinc ~ .,data=xy)",
#    regPredName='predictions')


# TODO:  (a) classification case, (b) only 1 X variable, (c) factors to
# numeric conversion of factors in new obs

# provide a simple, fast alternative to deep learning, random forests,
# XGBoost etc.

# it breaks the X space into patches defined by k-means clustering; a
# model, say mean or conditional mean (lm, glm) is fit in each cluster;
# to predict a new data point, determine its cluster and then use the
# predict() function there

# the per-cluster regression/classification model is specified via the
# regCall argument; for convenience, it can be specified as 'uncond',
# 'lm' or 'glm'; 'uncond' means average if Y is numeric, and means the
# most common value of Y in the classification case

# requirements

# 1. the fitting function must have a 'data' argument, as with 'lm' and 'ranger'
# 
# 2. the return value of the fitting function must have a 'predict' method
# 
# 3. no character or logical variables

patchReg <- function(XYdata,yName,numClust,regCall,
   savePreds=TRUE,regPredFtn=NULL,classPredFtn=NULL,
   holdout = floor(min(1000,0.1*nrow(XYdata)))) 
{

   yCol <- which(names(XYdata) == yName)
   Xdata <- XYdata[,-yCol,drop=FALSE]
   Ydata <- XYdata[,yCol]
   classif <- class(Ydata) == 'factor'
   nYlvls <- length(levels(Ydata))
   binClass <- (classif && nYlvls==2)
   multClass <- (classif && nYlvls==2)
   
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

   # adjust regCall if needed; xy is the dataset newData belonging 
   # to a given cluster
   if (regCall %in% c('uncond','lm','glm')) {  # shortcut specs
      if (regCall == 'uncond') {
         if (!classif)  # uncond, Y is numeric { 
            regCall <- 'function(xy) lm(wageinc ~ 1,xy)' 
         } else {  # uncond, Y a factor
            regCall <- 'function(xy) {y <- xy[,yCol]; tbl <- table(y);
               names(tbl)[which.max(tbl)' 
         } 
   } else if (regCall == 'lm') { 
             regCall <- paste( 'function(xy)',yName,'~ .,xy)') 
          } else {   # glm } 
          }
      }
   }

   # do separate fits to the clusters
   pROut <- lapply(clustData,evalr(regCall))
   # element i of pROut is the value returned by calling regCall on
   # cluster i
   pROut$centers <- kmeansOut$centers
   pROut$clustNums <- clustInfo$clustNums
   pROut$factorsInfo <- factorsInfo
   pROut$classif <- classif
   pROut$tstIdxs <- tstIdxs
   pROut$regPredFtn <- regPredFtn
   pROut$classPredFtn <- classPredFtn
   pROut$classNames <- levels(Ydata)
   class(pROut) <- 'prout'

   if (holdout != 0) {
      preds <- predict(pROut,tstXData)
      if (!classif)
         pROut$testAcc <- mean(abs(tstYData[,1] - preds))
      else 
         pROut$testAcc <- mean(tstYData[,1] != preds)
      if (savePreds) {
         pROut$preds <- preds
      }
   }

   pROut
}

plot.prout <- function(object) 
{
   preds <- object$preds
   clustNums <- z$clustNums[object$tstIdxs]
   dta <- data.frame(preds=preds,clustNums=clustNums)
   densityplot(~preds,groups=clustNums,data=dta,plot.points=FALSE,auto.key=TRUE)
}

print.prout <- function(object) 
{
   cat('\ntestAcc:',object$testAcc,'\n\n')
   cat('\ncenters:\n\n')
   print(object$centers)
   if (!is.null(object[[1]]$coefficients)) {
      cat('\ncoefficients:\n\n')
      for (i in 1:nrow(object$centers)) {
         # print(object[[i]]$coefficients)
         print(coef(object[[i]]))
      }
   }
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
      newx <- newX[i,,drop=FALSE]
      closestIdx <- FNN::get.knnx(object$centers,newx,k=1)$nn.index
      tmp <- predict(object[[closestIdx]],newx)
      if (object$classif) {
         classPredFtn <- object$classPredFtn
         tmp <- classPredFtn(tmp)
         if (substr(tmp,1,4) == 'dfr.')  # due to factorsToDummies
            tmp <- substr(tmp,5,nchar(tmp))
      } else {  # regression case
         regPredFtn <- object$regPredFtn
         if (!is.null(regPredFtn)) 
            # tmp <- tmp[[regPredName]]
            tmp <- regPredFtn(tmp)
      }
      preds[i] <- tmp
   }
   if (inherits(preds,'list')) preds <- unlist(preds)  # qeKNN
   preds
}

classPredFtn.qeML <- function(tmp) tmp <- tmp[['predClasses']]

regPredFtn.ranger <- function(tmp) tmp <- tmp[['predictions']]
classPredFtn.ranger <- function(tmp,object) {
   tmp <- tmp[['predictions']]
   tmp <- levels(tmp)[tmp]
}

# examples

# library(qeML)
# patchReg(svcensus,"wageinc",4,"function(xy) lm(wageinc~.,data=xy)")
# patchReg(svcensus,"gender",2,"function(xy) qeLogit(xy,'gender',
#    yesYVal='female')",classPredFtn=classPredFtn.qeML)
# patchReg(svcensus,"wageinc",8,"function(xy) qeKNN(xy,'wageinc',k=100)")
# patchReg(svcensus,"gender",2,"function(xy) qeKNN(xy,'gender',k=100,
#    yesYVal='female')",classPredFtn=classPredFtn.qeML)
# patchReg(svcensus,"occ",2,"function(xy) qeKNN(xy,'occ',k=100)",
#   classPredFtn=classPredFtn.qeML)
# patchReg(svcensus,"gender",2,"function(xy) qeLASSO(xy,'gender')",
#   classPredFtn=classPredFtn.qeML)
# library(ranger)
# patchReg(svcensus,"wageinc",2,"function(xy) ranger(wageinc ~ .,data=xy)",
#    regPredFtn=regPredFtn.ranger)
# patchReg(svcensus,"gender",2,"function(xy) ranger(gender ~ .,data=xy)",
#    classPredFtn=classPredFtn.ranger)
# replicMeans(500,"patchReg(svcensus,\"wageinc\",2,\"function(xy) lm(wageinc ~ .,data=xy)\")$testAcc")

require(caret)
require(reticulate)
require(xgboost)

args <- commandArgs(trailingOnly = TRUE)
inputNumpyPath <- args[1]
outputDirPath <- args[2]

set.seed(23489)

np <- import("numpy")
loaded <- np$load(inputNumpyPath)
features = loaded['features']
N <- nrow(features)
M <- ncol(features)
#N <- 50000 # small set for debuging
labels = loaded['labels']
validationIndicator = loaded['validationIndicator']

features <- features[1:N,]
labels <- labels[1:N,]
validationIndicator <- validationIndicator[1:N]

loaded <- NULL
ds <- NULL

macro.recall <- function (data,
                          lev = NULL,
                          model = NULL) {
  clsLevels <- levels(data$obs)
  clsCount <- length(levels(data$obs))
  recalls <- c()
  for(lvl in clsLevels) {
    obsPos <- data$obs == lvl
    predPos <- data$pred == lvl
    tp <- sum(obsPos & predPos)
    totPos <- sum(obsPos)
    if(totPos>0){
      recall <- tp/totPos
      recalls <- c(recalls, recall)
      #print(paste0('level ',lvl,' tp ',tp,' totP ',totPos,' recall ',recall))
    }
  }
  res <- mean(recalls)
  #print(paste0('Macro recall is ',res))
  names(res) <- "MacroRecall"
  return(res)
}

fit_control <- trainControl(
  verboseIter = TRUE,
  #sampling="down",
  classProbs= TRUE,
  index=list(which(!validationIndicator)),
  #indexFinal=which(!validationIndicator),
  summaryFunction = macro.recall,
  allowParallel=T)

for(classificationIdx in 1:3) {
  classificationName <- switch(classificationIdx,'root','vowel','consonant')
  outPrefix = file.path(outputDirPath,classificationName)
  target <- labels[,classificationIdx]

  L <- length(unique(target))
  
  if(is.null(ds)) {
    ds <- data.frame(matrix(nrow=N,ncol=M+1))
    ds[,1:M] <- features
  }
  ds[,M+1] <- factor(target, levels=0:(L-1), labels=c(paste0('cls',0:(L-1))))
  names(ds) <- c(paste0("feat",1:M),"target")
  
  features <- NULL


  fitRes <- train(
    target ~ . , 
    data=ds,
    metric='MacroRecall',
    trControl = fit_control,
    method = "xgbTree")
  for(toFile in c(T,F)) {
    if(toFile) {
      statsFile = paste0(outPrefix,".summary.txt")
      sink(statsFile)
    } else
      sink()
    print(fitRes)
    sink()
  }
  xgbOut = paste0(outPrefix,".xgboost.model")
  xgb.save(fitRes$finalModel, xgbOut)
  fitResOut = paste0(outPrefix,".fitResult.RData")
  save(fitRes,file=fitResOut)
}
print("Done")
  


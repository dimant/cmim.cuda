cmim <- function(...) UseMethod("cmim")

classImbalance <- function(y) {
  counts <- table(y)
  ci <- 100 * (max(counts) - min(counts))/sum(counts)
  ci
}

prepareData <- function(x, y, na.rm=TRUE) {
    d <- cbind(x, y)

    classes <- length(unique(y))
    ci <- classImbalance(y)
    maxBits <-  sum(ceiling(sort(sapply(sapply(lapply(d, unique), length), log2), decreasing=TRUE))[1:5])

    if(maxBits > 31)
        stop("Data is not suitable for the CMIM algorithm because the 5 widest dimensions require more than 31bits\n")

    if(na.rm)
        d <- na.omit(d)

    n <- dim(d)[1]
    p <- dim(d)[2]

    d <- as.character(as.vector(as.matrix(t(d))))

    result <- list(data=d,
                   n=n,
                   p=p,
                   classes=classes,
                   classImbalance=ci,
                   maxBits=maxBits)

    class(result) <- "cmimData"

    result
}

print.cmimData <- function(x, ...) {
    cat(sprintf("Observations: %d\n", x$n))
    cat(sprintf("Dimensions: %d\n", x$p))
    cat(sprintf("Classes: %d\n", x$classes))
    cat(sprintf("Class Imbalance: %f\n", x$classImbalance))
    cat(sprintf("Bits required for 5 widest dimensions: %d\n\n", x$maxBits))
}

checkHardwareSupport <- function() {
  name <- ""
  memory <- integer(1)
  major <- integer(1)
  minor <- integer(1)
  status <- ""
  result <- .C("r_HardwareSupport", name, memory, major, minor, status, PACKAGE="libbfscuda")

  if(result[[5]] == "OK") {
    cat("Device Type:     ", result[[1]], sep="")
    cat("\nMemory:          ", result[[2]] / (1024^3), "GB", sep="")
    cat("\nCUDA capability: ", result[[3]], ".", result[[4]], "\n", sep="")

    TRUE
  } else {
    FALSE
  }
}

cmim.default <- function(x, y, k=2, max.parallelism=0, na.rm=TRUE, ...) {
    result <- list()


    d <- prepareData(x, y, na.rm)

    result <- .C("r_BFSCuda",
                 data=as.character(d$data),
                 n=as.integer(d$n),
                 p=as.integer(d$p),
                 k=as.integer(k),
                 max.parallelism=as.integer(max.parallelism),
                 weights=double(d$p-1),
                 indices=double(d$p-1),
                 PACKAGE="libbfscuda")
    
    class(result) <- "cmim"
    result$call <- match.call()
    result
}

cmim.formula <- function(formula, data, k=2, max.parallelism=0, na.rm=TRUE, ...) {
    mf <- model.frame(formula=formula, data=data)
    x <- model.matrix(attr(mf, "terms"), data=mf)
    y <- model.response(mf)

    result <- cmim.default(x, y, ...)
    result$call <- match.call()
    result$formula <- formula

    result
}

mifs <- function(x, y) {
  d <- cbind(x, y)
  
  weights <- as.matrix(sapply(1:ncol(x), function(v) {
    mi.empirical(table(d[c(v, ncol(d))]))
  }))
  rownames(weights) <- colnames(d[1:length(d) - 1])

  weights
}

mifsuk <- function(x, y, k=1, max.parallelism=0, na.rm=FALSE, ...) {
  d <- prepareData(x, y, na.rm)
  terms <- mifs(x,y)

  for(i in 1:k) {
    result <- list()
    result <- .C("r_mifsuk",
                 data=as.character(d$data),
                 n=as.integer(d$n),
                 p=as.integer(d$p),
                 k=as.integer(i),
                 max.parallelism=as.integer(max.parallelism),
                 weights=double(d$p-1),
                 PACKAGE="libbfscuda")
    
    terms <- cbind(terms, result$weights)
  }

  terms
}

cmifsExact <- function(x, y, k=1, max.parallelism=0, na.rm=FALSE, ...) {
  infgain <- mifs(x, y)
  d <- prepareData(x, y, na.rm)
  
  result <- .C("r_cmifs_exact",
               data=as.character(d$data),
               n=as.integer(d$n),
               p=as.integer(d$p),
               k=as.integer(2),
               max.parallelism=as.integer(max.parallelism),
               infgain=infgain,
               weights=double(d$p-1),
               indices=double(d$p-1),
               PACKAGE="libbfscuda")

  result
}

mifsmed <- function(x, y, k=1, max.parallelism=0, na.rm=FALSE, ...) {
  result <- list()
  
  d <- prepareData(x, y, na.rm)
  terms <- mifs(x, y)

  for(i in 1:k) {
    result <- .C("r_mifsmed",
                 data=as.character(d$data),
                 n=as.integer(d$n),
                 p=as.integer(d$p),
                 k=as.integer(i),
                 max.parallelism=as.integer(max.parallelism),
                 weights=double(d$p-1),
                 PACKAGE="libbfscuda")
    terms <- cbind(terms, result$weights)
  }
  
  terms
}

mRRmatrix <- function(x, y, max.parallelism=0, na.rm=TRUE, diag=FALSE, upper=FALSE) {
  result <- list()

  d <- prepareData(x, y, na.rm)

  result <- .C("r_mRRmatrix",
               data=as.character(d$data),
               n=as.integer(d$n),
               p=as.integer(d$p),
               max.parallelism=as.integer(max.parallelism),
               weights=double((d$p-1)^2),
               PACKAGE="libbfscuda")

#  result <- as.dist(matrix(result$weights, ncol=d$p-1, nrow=d$p-1))
#  attr(result, "call") <- match.call()

  result <- matrix(result$weights, ncol=d$p-1, nrow=d$p-1)
  
  result
}

#mRRspan <- function(x, y, max.parallelism=0, na.rm=TRUE, ...) {
#  result <- list()
# 
#  result <- mRRmatrix(x, y, max.parallelism, na.rm)
#  myspantree <- mst(result$weights) * result$weights
# 
#  result$call <- match.call()
#  result$weights <- apply(myspantree, 2, min)
#  result$indices <- rev(order(order(result$weights)))
# 
#  result
#}
# 
#mRRshortpaths <- function(x, y, max.parallelism=0, na.rm=TRUE) {
#  result <- mRRmatrix(x, y, max.parallelism, na.rm)
#  paths <- allShortestPaths(result$weights)
# 
#  # add up the weights along a path recursively
#  f <- function(x, l) {
#    if(length(x) > 1) {
#      f(x[2:length(x)], l + result$weights[x[1],x[2]])
#    } else {
#      l
#    }
#  }
#  
#  pathlengths <- apply(expand.grid(1:ncol(x), 1:ncol(x)), 1, function(x) {
#    f(extractPath(paths, x[1], x[2]), 0)
#  } )
# 
#  weights <- apply(matrix(pathlengths, ncol=ncol(x)), 2, sum)
# 
#  result$weights <- weights
#  result$indices <- rev(order(order(weights)))
#  result
#}

predict.cmim <- function(object, newDataNULL, ...) {

}

print.cmim <- function(x, ...) {
    cat("Conditional Mutual Information Estimates:\n")
    indexed <- rev(order(x$weights))
    
    cat("Var\tWeight\n")
    for(i in 1:length(indexed)) {
      cat(x$indices[indexed[i]],"\t",x$weights[indexed[i]],"\n")
    }
    
}

summary.cmim <- function(object, ...) {
    indexed <- rev(order(object$weights))

    cat("Call: ")
    print(object$call)
    cat("\n")
    
    cat("\t Var\t Weight\n")
    cat("Best\t", object$indices[indexed[1]], "\t", object$weights[indexed[1]], "\n")
    cat("Worst\t", object$indices[indexed[length(indexed)]], "\t", object$weights[indexed[length(indexed)]],"\n")
    cat("\n")
    cudaSupport <- checkHardwareSupport()
}

plot.cmim <- function(x, names.arg = c(), ...) {
    indexed <- rev(order(x$weights))

    if(length(names.arg) == 0) {
      names.arg = x$indices[indexed]
    }
    
    barplot(x$weights[indexed], names.arg = names.arg, ylab="Weight", xlab="Variable")
}

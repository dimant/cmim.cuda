\name{cmim}
%- Also NEED an '\alias' for EACH other topic documented here.
\alias{cmim}
\alias{cmim.formula}
\alias{cmim.default}
\alias{print.cmim}
\alias{plot.cmim}
\alias{summary.cmim}
\title{
Conditional Mutual Information Maximisation.
}
\description{
Estimates the maximal conditional mutual information between a class and
a dimension conditional on any k other dimensions.
}

\usage{
cmim(...)
\method{cmim}{default}(x, y, k=2, max.parallelism=0, na.rm=TRUE, ...)
\method{cmim}{formula}(formula, data, k=2, max.parallelism=0, na.rm=TRUE, ...)
\method{print}{cmim}(x,...)
\method{plot}{cmim}(x, names.arg, ...)
\method{summary}{cmim}(object,...)

}


\arguments{
  \item{x}{
    A vector describing the class for each row.
  }

  \item{y}{
    A matrix of variables.
  }

  \item{formula}{
    A formula, such as y ~ x, where y is vector
    describing the class for each row and x is a matrix of
    variables
  }

  \item{data}{
    A data.frame (or list) from which the variables in the formula should be taken.
  }

  \item{k}{
    the maximal number of conditional variables to consider when
    estimating I(Y ; X_i | X_u, X_v, X_w,...). Note that increasing this
    value will make the algorithm run a lot slowlier (hours instead of
    minutes) and will perform worse unless there is enough data to
    estimate k + 2 dimensional probability functions.
  }

  \item{max.parallelism}{
    number of parallel executions of the algorithm.
  }

  \item{na.rm}{
    if TRUE rows containing missing values will be omitted. If FALSE,
    missing values will be treated as a value.
  }

  \item{object}{
    An object of type \code{cmim} as returned by \code{cmim()}.
  }

  \item{names.arg}{
    A list of names of the variables to be used when plotting.
  }

  \item{...}{
    currently disregarded.
  }

}

\value{
  An object of class \code{cmim}. The class is represented as a list
  including the elements

  \item{data}{The data after preprocessing for transfer to GPU memory.}
  \item{n}{number of rows}
  \item{p}{number of columns}
  \item{k}{number of conditional variables taken into consideration}
  \item{max.parallelism}{number of iterations evaluated in parallel}
  \item{weights}{a list where each element contains the conditional
  mutual information of the variable the index of which is contained in the
  indices list at the corresponding position}
  \item{indices}{the first element contains the index of the variable
  the weight of which is contained at the first position in the weights list}
  \item{call}{The invocation of the cmim method.}  
}

\details{
%%  ~~ If necessary, more details than the description above ~~
Warning
Be mindful when changing the default values for the parameters k and
max.parallelism.

Increasing k will result in significantly longer
running times (hours instead of minutes).

Increasing max.parallelism may result in unpredictable behaviour,
crashing R or even rendering your system unstable. The reason for this
is that a larger number of parallel executions require a larger amount
of memory on the graphics card. If the same graphics card is used for both,
computation and display, both tasks will compete for memory.
}

%%\references{
%% ~put references to the literature/web site here ~
%%}

\author{
Diman Todorov
}

\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

}
%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

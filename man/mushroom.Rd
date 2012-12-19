\name{mushroom}
\alias{mushroom}
\docType{data}
\title{
Mushroom
}
\description{
This data set includes descriptions of hypothetical samples
corresponding to 23 species of gilled mushrooms in the Agaricus and
Lepiota Family (pp. 500-525).  Each species is identified as
definitely edible, definitely poisonous, or of unknown edibility and
not recommended.  This latter class was combined with the poisonous
one.  The Guide clearly states that there is no simple rule for
determining the edibility of a mushroom; no rule like ``leaflets
three, let it be'' for Poisonous Oak and Ivy.
}
\usage{data(mushroom)}
\format{
  A data frame with 8124 observations on 23 variables.
}
\source{
http://archive.ics.uci.edu/ml/datasets/Mushroom
}
\references{
Schlimmer,J.S. (1987). Concept Acquisition Through Representational Adjustment (Technical Report 87-19). Doctoral disseration, Department of Information and Computer Science, University of California, Irvine.

Iba,W., Wogulis,J., & Langley,P. (1988). Trading off Simplicity and Coverage in Incremental Concept Learning. In Proceedings of the 5th International Conference on Machine Learning, 73-79. Ann Arbor, Michigan: Morgan Kaufmann.

Duch W, Adamczak R, Grabczewski K (1996) Extraction of logical rules from training data using backpropagation networks, in: Proc. of the The 1st Online Workshop on Soft Computing, 19-30.Aug.1996, pp. 25-30, [Web Link]

Duch W, Adamczak R, Grabczewski K, Ishikawa M, Ueda H, Extraction of crisp logical rules using constrained backpropagation networks - comparison of two new approaches, in: Proc. of the European Symposium on Artificial Neural Networks (ESANN'97), Bruge, Belgium 16-18.4.1997. 
}
\examples{
data(mushroom)
## maybe str(mushroom) ; plot(mushroom) ...
}
\keyword{datasets}

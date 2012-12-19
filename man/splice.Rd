\name{splice}
\alias{splice}
\docType{data}
\title{
Splice
}
\description{
This data was collected to analyse splice-junction gene sequences. When proteins are formed
from DNA strands, splice junctions are the points at which DNA is removed. The challenge in 
this data is to determine the boundaries between DNA
which is retained and DNA which is removed after splicing. The data, which consists of DNA sub-sequences,
is labelled in three classes:
a boundary between retained and removed DNA (EI), boundary between removed and retained DNA (IE), and
spurious sequences (N).

  
}
\usage{data(splice)}
\format{
  A data frame with 3190 observations on 61 variables.
}
\source{
http://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29
}
\references{


M. O. Noordewier and G. G. Towell and J. W. Shavlik, 1991; Training Knowledge-Based Neural Networks to Recognize Genes in DNA Sequences. Advances in Neural Information Processing Systems, volume 3, Morgan Kaufmann.

G. G. Towell and J. W. Shavlik and M. W. Craven, 1991; Constructive Induction in Knowledge-Based Neural Networks. In Proceedings of the Eighth International Machine Learning Workshop, Morgan Kaufmann.

G. G. Towell, 1991; Symbolic Knowledge and Neural Networks: Insertion, Refinement, and Extraction. PhD Thesis, University of Wisconsin - Madison.

G. G. Towell and J. W. Shavlik, 1992; Interpretation of Artificial Neural Networks: Mapping Knowledge-based Neural Networks into Rules. In Advances in Neural Information Processing Systems, volume 4, Morgan Kaufmann.
}
\examples{
data(splice)
## maybe str(splice) ; plot(splice) ...
}
\keyword{datasets}

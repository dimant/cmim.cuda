\name{krvskp}
\alias{krvskp}
\docType{data}
\title{
King Rook VS King Pawn
}
\description{
The chess endgame king rook vs king pawn is
fundamental as many other endgames can be reduced to this one.
The data is composed of instances of the king rook versus king pawn setup
37 binary attributes describe an instance of the end game and the
class signifies whether white can win - no overlap between classes.
}
\usage{data(krvskp)}
\format{
  A data frame with 3196 observations on 37 variables.
}
\source{
http://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29
}
\references{
Alen D. Shapiro (1983,1987), Structured Induction in Expert Systems. Addison-Wesley. This book is based on Shapiro's Ph.D. thesis (1983) at the University of Edinburgh entitled "The Role of Structured Induction in Expert Systems".

Stephen Muggleton (1987), Structuring Knowledge by Asking Questions. pp.218-229 in Progress in Machine Learning edited by I. Bratko and Nada Lavrac, Sigma Press, Wilmslow, England SK9 5BB.

Robert C. Holte, Liane Acker, and Bruce W. Porter (1989), Concept Learning and the Problem of Small Disjuncts. Proceedings of IJCAI. Also available as technical report AI89-106, Computer Sciences Department, University of Texas at Austin, Austin, Texas 78712.
}
\examples{
data(krvskp)
## maybe str(krvskp) ; plot(krvskp) ...
}
\keyword{datasets}

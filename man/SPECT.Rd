\name{SPECT}
\alias{SPECT}
\docType{data}
\title{
SPECT
}
\description{
SPECT imaging is used as a diagnostic tool for myocardial perfusion. The patient is injected with
radioactive tracer (in our case Tl-201). Then two studies are performed, one 10-15 min. after
injection during maximal stress - called stress study (stress image), and one 2-5 hours after
injection - called rest study (rest image). The studies are collected as two sets of three-dimensional
images. All the images represent LV muscle perfusion that is proportional to distribution of
radioactive counts within the myocardium [10]. Cardiologists compare stress and rest studies in
order to detect abnormalities in the LV perfusion.
}
\usage{data(SPECT)}
\format{
  A data frame with 266 observations on 23 variables.
}
\source{
http://archive.ics.uci.edu/ml/datasets/SPECT+Heart
}
\references{
Kurgan, L.A., Cios, K.J., Tadeusiewicz, R., Ogiela, M. & Goodenday, L.S. Knowledge Discovery Approach to Automated Cardiac SPECT Diagnosis. Artificial Intelligence in Medicine, vol. 23:2, pp 149-169, Oct 2001

Cios, K.J., Wedding, D.K. & Liu, N. CLIP3: cover learning using integer programming. Kybernetes, 26:4-5, pp 513-536, 1997

Cios K. J. & Kurgan L. Hybrid Inductive Machine Learning: An Overview of CLIP Algorithms, In: Jain L.C., and Kacprzyk J. (Eds). New Learning Paradigms in Soft Computing, Physica-Verlag (Springer), 2001
}
\examples{
data(SPECT)
}
\keyword{datasets}

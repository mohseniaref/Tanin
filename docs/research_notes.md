# Research notes

Amiri-Simkooei's framework separates the functional model (deterministic terms)
from the stochastic model (noise covariance). Candidate harmonics augment the
design matrix with sine/cosine columns; null and alternative models are compared
with residual tests. The paper emphasizes estimating white/flicker/random-walk
noise components before interpreting significance, and describes LS-VCE for that
purpose. Tanin's MVP implements the nested-model F test and accepts a covariance
path internally, but does not yet estimate colored-noise components.

LSSA is least-squares fitting over a supplied frequency grid and can include a
trend. Lomb–Scargle is the specialized floating-mean sinusoid periodogram for
uneven observations; SciPy uses angular frequencies, whereas Astropy uses cycles
per unit time. Their normalizations are not interchangeable. A spectral maximum
is exploratory and is not a multiple-testing-corrected detection.

The LSWAVE repository and pytrf were consulted as architectural references only;
Tanin reimplements the small MVP independently and does not depend on either.
MintPy's current time-series convention is an HDF5 `/date` vector, a
`/timeseries` array shaped `(time, length, width)`, and root metadata. The adapter
keeps metadata and maps axes to `time, y, x` without making MintPy mandatory.

Provenance: equations are expressed as standard weighted least-squares harmonic
regression, with references documented here and in public API docstrings. The
implementation is original and does not copy source code.

The LS-VCE extension follows the matrix equations in the local 2007/2008
references. The optional non-negative solver is included because unconstrained
variance-component estimates can be negative. The flicker model is a finite-band
1/f quadrature approximation and should be validated against domain-specific
simulations before production use.

The expanded LS-VCE workflow now includes a numeric time-series reader, public
functional design-matrix utilities, a generic finite-band power-law covariance,
component covariance/standard-error estimates, w-statistics, and iteration
diagnostics. These are independent reproductions of the published equations;
the local workspace contained the papers but not the Springer supplementary
source archive, so no claim of byte-for-byte supplementary-code compatibility
is made.

Primary references:

- Amiri-Simkooei, A. R., Tiberius, C. C. J. M., & Teunissen, P. J. G. (2007),
  *Assessment of noise in GPS coordinate time series: Methodology and results*,
  JGR Solid Earth 112, B07413, doi:10.1029/2006JB004913.
- Teunissen, P. J. G., & Amiri-Simkooei, A. R. (2008), *Least-squares variance
  component estimation*, Journal of Geodesy 82, 65–82,
  doi:10.1007/s00190-007-0157-x.
- Amiri-Simkooei, A. R. (2007), *Least-squares variance component estimation:
  Theory and GPS applications*, PhD thesis, Delft University of Technology,
  Publication on Geodesy 64.
- Amiri-Simkooei, A. R. (2013), *On the nature of GPS draconitic year periodic
  pattern in multivariate position time series*, JGR Solid Earth 118,
  2500–2511, doi:10.1002/jgrb.50199.
- Amiri-Simkooei, A. R. (2016), *Non-negative least-squares variance component
  estimation with application to GPS time series*, Journal of Geodesy 90,
  451–466, doi:10.1007/s00190-016-0886-9.
- Amiri-Simkooei, A. R., Hosseini-Asl, M., Asgari, J., & Zangeneh-Nejad, F.
  (2019), *Offset detection in GPS position time series using multivariate
  analysis*, GPS Solutions 23, 13, doi:10.1007/s10291-018-0805-z.
- Ghaderpour, E., Antonielli, B., Bozzano, F., Scarascia Mugnozza, G., &
  Mazzanti, P. (2024), *A fast and robust method for detecting trend turning
  points in InSAR displacement time series*, Computers & Geosciences 185,
  105546, doi:10.1016/j.cageo.2024.105546.

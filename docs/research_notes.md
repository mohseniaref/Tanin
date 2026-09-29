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

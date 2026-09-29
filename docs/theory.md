# Tanin theory and equations

## Functional model

For an observation vector `y` at times `t`, Tanin uses a linear functional
model:

\[
\mathbf y = \mathbf A\boldsymbol\beta + \mathbf e.
\]

For an intercept, linear trend, and `q` harmonic periods:

\[
y(t)=c+v(t-t_0)+
\sum_{k=1}^{q}\left[a_k\cos(\omega_k t)+b_k\sin(\omega_k t)\right]+e(t),
\]

where:

\[
\omega_k=\frac{2\pi}{P_k}.
\]

The trend is centered in time to reduce correlation between intercept and
slope. Tanin uses decimal days internally for datetime coordinates.

## Harmonic amplitude and phase

For one harmonic:

\[
a\cos(\omega t)+b\sin(\omega t)
=A\cos(\omega t+\phi).
\]

The conversion used by Tanin is:

\[
A=\sqrt{a^2+b^2},
\qquad
\phi=\operatorname{atan2}(-b,a).
\]

This phase convention is tested and recorded in the output metadata.

## Weighted and generalized least squares

With independent observation weights `W`:

\[
\hat{\boldsymbol\beta}
= (\mathbf A^T\mathbf W\mathbf A)^{-1}
\mathbf A^T\mathbf W\mathbf y.
\]

With a full covariance matrix `C`, Tanin whitens using a Cholesky factorization
where possible:

\[
\mathbf C=\mathbf L\mathbf L^T,
\qquad
\mathbf L^{-1}\mathbf y
=\mathbf L^{-1}\mathbf A\boldsymbol\beta
 + \mathbf L^{-1}\mathbf e.
\]

No explicit matrix inverse is used for the least-squares solve.

## Specified-period testing

The null and alternative models are nested:

\[
H_0: \mathbf y=\mathbf A_0\boldsymbol\beta_0+\mathbf e,
\]

\[
H_1: \mathbf y=\mathbf A_1\boldsymbol\beta_1+\mathbf e,
\]

where `A1` adds the cosine and sine columns for the candidate period. The
nested-model statistic is:

\[
F=\frac{(RSS_0-RSS_1)/r}{RSS_1/(n-p_1)},
\]

where `r=2` added harmonic parameters, `n` is the number of valid
observations, and `p1` is the number of alternative-model parameters.

The test asks whether the two harmonic coefficients are jointly zero. A
non-significant result means “the data did not establish a signal at this
period”; it does not prove that the physical process is absent.

## LS-VCE stochastic model

LS-VCE represents the observation covariance as:

\[
\mathbf Q_y=\mathbf Q_0+
\sum_{k=1}^{p}\sigma_k\mathbf Q_k,
\]

where `Qk` are known cofactor matrices and `sigma_k` are unknown variance
components. Tanin supports white, flicker-like, random-walk, and finite-band
power-law cofactors.

The power-law spectrum is represented conceptually as:

\[
P_y(f)=P_0\left(\frac{f}{f_0}\right)^\kappa.
\]

The standard special cases are approximately:

```text
kappa =  0   white noise
kappa = -1   flicker noise
kappa = -2   random-walk-like noise
```

For a current covariance estimate `Qy`, the generalized least-squares
projector used in the LS-VCE equations is:

\[
\mathbf P=
\mathbf Q_y^{-1}-
\mathbf Q_y^{-1}\mathbf A
(\mathbf A^T\mathbf Q_y^{-1}\mathbf A)^{-1}
\mathbf A^T\mathbf Q_y^{-1}.
\]

The practical normal matrix and right-hand side used by Tanin are:

\[
N_{kl}=\frac12\operatorname{tr}
(\mathbf P\mathbf Q_k\mathbf P\mathbf Q_l),
\]

\[
l_k=\frac12\mathbf v^T\mathbf P\mathbf Q_k\mathbf P\mathbf v,
\]

with residual vector `v`. The unconstrained update is:

\[
\hat{\boldsymbol\sigma}=\mathbf N^{-1}\mathbf l.
\]

The non-negative option solves the corresponding bounded least-squares problem
with `sigma_k >= 0`.

## Component uncertainty and w-statistics

Under the LS-VCE approximation, the component-estimator covariance is:

\[
\mathbf Q_{\hat\sigma}\approx\mathbf N^{-1}.
\]

Tanin reports:

\[
SE(\hat\sigma_k)=
\sqrt{[\mathbf Q_{\hat\sigma}]_{kk}},
\]

and the diagnostic statistic:

\[
w_k=\frac{\hat\sigma_k}{SE(\hat\sigma_k)}.
\]

These diagnostics are approximate when non-negativity constraints are active,
when the covariance components are poorly identifiable, or when the functional
model is misspecified.

## Uncertainty of amplitude and phase

For the coefficient covariance submatrix of `(a,b)`, Tanin uses the delta
method. With:

\[
g_A=\left(\frac{a}{A},\frac{b}{A}\right),
\qquad
g_\phi=\left(\frac{b}{A^2},-\frac{a}{A^2}\right),
\]

the propagated variances are:

\[
\operatorname{var}(A)=g_A\mathbf Q_{ab}g_A^T,
\qquad
\operatorname{var}(\phi)=g_\phi\mathbf Q_{ab}g_\phi^T.
\]

## Interpretation for InSAR

For an InSAR cube:

- Estimate or select a stochastic model before interpreting weak seasonal terms.
- Include known offsets and discontinuities in the functional model.
- Do not force annual or semiannual coefficients to zero before testing them.
- Use covariance-aware tests when residuals are temporally correlated.
- Correct for multiple comparisons when producing significance maps.
- Treat a non-significant term as “not detected,” not as proof of zero physical
  deformation.
- Check reference date, reference pixel, units, masks, and spatial correlation.

## References

- Amiri-Simkooei, A. R., Tiberius, C. C. J. M., & Teunissen, P. J. G. (2007).
  Assessment of noise in GPS coordinate time series: Methodology and results.
  *Journal of Geophysical Research: Solid Earth*, 112, B07413.
  https://doi.org/10.1029/2006JB004913
- Teunissen, P. J. G., & Amiri-Simkooei, A. R. (2008). Least-squares variance
  component estimation. *Journal of Geodesy*, 82, 65–82.
  https://doi.org/10.1007/s00190-007-0157-x
- Amiri-Simkooei, A. R. (2007). *Least-squares variance component estimation:
  Theory and GPS applications*. PhD thesis, Delft University of Technology,
  Publication on Geodesy 64, Netherlands Geodetic Commission.
- Amiri-Simkooei, A. R. (2013). On the nature of GPS draconitic year periodic
  pattern in multivariate position time series. *Journal of Geophysical
  Research: Solid Earth*, 118, 2500–2511.
  https://doi.org/10.1002/jgrb.50199
- Amiri-Simkooei, A. R. (2016). Non-negative least-squares variance component
  estimation with application to GPS time series. *Journal of Geodesy*, 90,
  451–466. https://doi.org/10.1007/s00190-016-0886-9
- Amiri-Simkooei, A. R., Hosseini-Asl, M., Asgari, J., & Zangeneh-Nejad, F.
  (2019). Offset detection in GPS position time series using multivariate
  analysis. *GPS Solutions*, 23, 13. https://doi.org/10.1007/s10291-018-0805-z
- Ghaderpour, E., Antonielli, B., Bozzano, F., Scarascia Mugnozza, G., &
  Mazzanti, P. (2024). A fast and robust method for detecting trend turning
  points in InSAR displacement time series. *Computers & Geosciences*, 185,
  105546. https://doi.org/10.1016/j.cageo.2024.105546

# Design

The core is one-dimensional numerical code (`time`, `models`, `harmonic`,
`spectrum`) wrapped over arbitrary xarray broadcast dimensions with
`apply_ufunc`. Results are xarray-native and preserve spatial coordinates. I/O
is isolated under `io`, with optional h5py-only MintPy support. Future work can
add covariance/noise estimation, offsets, uncertainty outputs, Dask chunk
planning, and a MintPy writer without coupling those concerns to regression.

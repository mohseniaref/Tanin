import numpy as np
from scipy.stats import f, chi2

def nested_f_test(rss0, rss1, n, added_parameters):
    df1 = int(added_parameters); df2 = n - df1
    if df2 <= 0 or rss1 <= 0: return np.nan, np.nan
    stat = ((rss0-rss1)/df1)/(rss1/df2)
    return float(stat), float(f.sf(max(stat, 0), df1, df2))

def chi_square(value, dof):
    return float(chi2.sf(value, dof))

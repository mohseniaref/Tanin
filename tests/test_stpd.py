import numpy as np
import tanin

def test_stpd_turning_points():
    t=np.arange(0.,300.,3.); y=np.where(t<100, .02*t, np.where(t<200, 2-.015*(t-100), .5+.025*(t-200)))
    result=tanin.stpd(t,y,max_turning_points=3,min_segment=8)
    assert len(result["turning_point_times"]) >= 2
    assert np.min(np.abs(result["turning_point_times"]-100)) < 15
    assert np.min(np.abs(result["turning_point_times"]-200)) < 15

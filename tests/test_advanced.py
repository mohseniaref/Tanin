import numpy as np
import tanin

def test_allssa_and_lswa():
    t=np.arange(0,1000,10.); y=np.cos(2*np.pi*t/365.25)+.3*np.cos(2*np.pi*t/180.)
    r=tanin.allssa(y,t,min_period=100,max_period=500,n_periods=100,n_harmonics=2)
    assert len(r["periods"])==2
    w=tanin.lswa(y,t,periods=(365.25,),window=400,step=50)
    assert w["amplitude"].shape[1]==1

def test_cross_jump_decompose_monitor():
    t=np.arange(100.); x=np.sin(t/10); y=2*x; y[50:]+=3
    cross=tanin.lscsa(x,y,t,periods=np.array([60.]))
    assert cross["coherence"][0] > .9
    jumps=tanin.detect_jumps(y,t,threshold=3)
    assert 50 in jumps["indices"]
    assert "remainder" in tanin.just_decompose(y,t,periods=(60,))
    assert "alarms" in tanin.just_monitor(y,t,window=10)
    wave = tanin.lscwa(x, y, t, periods=np.array([60.]))
    assert wave["cross_power"].shape == (1, len(t))

def test_joint_just_jump_detection():
    t=np.arange(0.,500.,5.); y=0.002*t+np.cos(2*np.pi*t/365.25); y[t>=245]+=2.
    result=tanin.just_jumps(y,t,periods=(365.25,),max_jumps=3,alpha=1e-5)
    assert np.min(np.abs(result["times"]-245)) < 10
    assert result["steps"][np.argmin(np.abs(result["times"]-245))] > 1

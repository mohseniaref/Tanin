import numpy as np
import matplotlib.pyplot as plt
import tanin

t=np.arange(0.,300.,3.)
y=np.where(t<100,.02*t,np.where(t<200,2-.015*(t-100),.5+.025*(t-200)))+np.random.default_rng(3).normal(0,.06,len(t))
r=tanin.stpd(t,y,max_turning_points=3,min_segment=8)
fig,ax=plt.subplots(figsize=(9,4.5))
ax.plot(t,y,'.',label='observations'); ax.plot(r['time'],r['trend'],lw=2,label='STPD connected trend')
for tp in r['turning_point_times']: ax.axvline(tp,color='tab:red',ls='--')
ax.set(xlabel='Days',ylabel='Signal',title='Sequential turning-point detection'); ax.legend(); fig.tight_layout(); fig.savefig('examples/tanin_stpd.png',dpi=160)
print('turning_point_times',r['turning_point_times']); print('plot examples/tanin_stpd.png')

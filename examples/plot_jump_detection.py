"""Plot joint JUST-style jump detection on a seasonal irregular series."""
import numpy as np
import matplotlib.pyplot as plt
import tanin

rng=np.random.default_rng(9); t=np.sort(rng.choice(np.arange(0.,1000.,6.),size=130,replace=False))
y=.001*t+1.2*np.cos(2*np.pi*t/365.25+.2)+rng.normal(0,.08,len(t)); y[t>=500]+=1.8; y[t>=780]-=1.1
r=tanin.just_jumps(y,t,periods=(365.25,),max_jumps=5,alpha=.001)
fig,ax=plt.subplots(2,1,figsize=(9,7),constrained_layout=True)
ax[0].plot(t,y,'.',label='observations'); ax[0].plot(t,r['fitted'],lw=2,label='joint trend + seasonal + steps')
for tau in r['times']: ax[0].axvline(tau,color='tab:red',ls='--',alpha=.7)
ax[0].set(xlabel='Days',ylabel='Signal',title='JUST-style joint jump detection'); ax[0].legend()
if len(r['times']): ax[1].bar(r['times'],r['steps'],width=12); ax[1].axhline(0,color='k',lw=.8)
ax[1].set(xlabel='Detected jump time (days)',ylabel='Step magnitude',title='Estimated offsets')
fig.savefig('examples/tanin_jump_detection.png',dpi=160)
print('detected_times',r['times']); print('estimated_steps',r['steps']); print('plot examples/tanin_jump_detection.png')

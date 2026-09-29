"""Diagnostic plots for ALLSSA, LSWA, LSCSA, LSCWA, and JUST utilities."""
import numpy as np
import matplotlib.pyplot as plt
import tanin

rng=np.random.default_rng(4); t=np.sort(rng.uniform(0,1200,180)); x=np.cos(2*np.pi*t/365.25)+.3*np.cos(2*np.pi*t/182.625)+rng.normal(0,.1,len(t)); y=1.8*x+rng.normal(0,.15,len(t));
a=tanin.allssa(x,t,100,500,300,2); w=tanin.lswa(x,t,(365.25,182.625),500,40); c=tanin.lscsa(x,y,t,np.geomspace(100,500,80)); cw=tanin.lscwa(x,y,t,np.geomspace(100,500,32)); d=tanin.just_decompose(x,t,(365.25,182.625)); m=tanin.just_monitor(x,t,30)
fig,ax=plt.subplots(3,2,figsize=(12,12),constrained_layout=True)
ax[0,0].plot(t,x,'.',ms=2); ax[0,0].set_title('Input irregular series')
ax[0,1].stem(a['periods'],a['amplitudes'],basefmt=' '); ax[0,1].set_title('ALLSSA selected periods')
im=ax[1,0].pcolormesh(w['time'],w['periods'],w['amplitude'].T,shading='auto'); ax[1,0].set_title('LSWA amplitude'); fig.colorbar(im,ax=ax[1,0])
ax[1,1].plot(c['period'],np.abs(c['cross_power'])); ax[1,1].set_title('LSCSA cross power')
im=ax[2,0].pcolormesh(cw['time'],cw['period'],cw['power'],shading='auto'); ax[2,0].set_title('LSCWA cross power'); fig.colorbar(im,ax=ax[2,0])
ax[2,1].plot(t,x,label='series'); ax[2,1].plot(t,d['seasonal']+d['trend'],label='JUST model'); ax[2,1].scatter(t[m['alarms']],x[m['alarms']],c='r',label='monitor alarms'); ax[2,1].legend(); ax[2,1].set_title('JUST decomposition/monitor')
for axy in ax.flat: axy.set_xlabel('time / period')
fig.savefig('examples/tanin_advanced_methods.png',dpi=160)
print('allssa periods',a['periods']); print('plot examples/tanin_advanced_methods.png')

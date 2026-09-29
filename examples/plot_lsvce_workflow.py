import numpy as np
import matplotlib.pyplot as plt
import tanin

rng=np.random.default_rng(22); t=np.arange(0.,12*365.25,12.); names=('white','flicker','random_walk'); truth=np.array([.04,.12,.0005]); q=tanin.covariance_components(t,names); C=sum(a*q[n] for a,n in zip(truth,names))+np.eye(len(t))*1e-8; y=.001*t+np.linalg.cholesky(C)@rng.normal(size=len(t))
r=tanin.estimate_noise_vce(y,t,components=names,max_iter=20)
estimated=np.array([r['components'][n] for n in names]); errors=np.array([r['standard_errors'][n] for n in names])
fig,ax=plt.subplots(2,1,figsize=(9,7),constrained_layout=True); ax[0].plot(t,y,'.',ms=2,label='observations'); ax[0].plot(t,.001*t,lw=2,label='design-model trend'); ax[0].legend(); ax[0].set_title('LS-VCE workflow input')
x=np.arange(len(names)); ax[1].bar(x-.18,truth,.36,label='simulated'); ax[1].bar(x+.18,estimated,.36,yerr=errors,capsize=4,label='LS-VCE ± 1σ'); ax[1].set_xticks(x); ax[1].set_xticklabels(names); ax[1].set_ylabel('Variance component'); ax[1].legend(); ax[1].set_title(f'Non-negative LS-VCE, {r["iterations"]} iterations')
fig.savefig('examples/tanin_lsvce_workflow.png',dpi=160); print('components',r['components']); print('standard_errors',r['standard_errors']); print('plot examples/tanin_lsvce_workflow.png')

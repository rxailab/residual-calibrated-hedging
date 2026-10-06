"""Reanalyse the archived pilot; no model execution or new simulator data.

Run with Python plus NumPy/Matplotlib. All bootstrap intervals are conditional
on one checkpoint and three fixed tasks, unadjusted for configuration selection.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'evidence' / 'pilot_records.json'
raw = json.loads(DATA.read_text(encoding='utf-8'))
records = raw['records']
TASKS = ['cartpole-swingup', 'walker-walk', 'cup-catch']
REF = {'cartpole-swingup': (880, 49), 'walker-walk': (985, 40), 'cup-catch': (981, 55)}
CONFIGS = [(8,4), (4,4), (2,4), (8,8), (4,8), (2,8)]
returns = np.zeros((6,3,5))
rows = []
for ci, (s,r) in enumerate(CONFIGS):
    chosen = [next(x for x in records if x['task']==t and x['eval_settings']['steps']==s and x['eval_settings']['replan_blocks']*4==r) for t in TASKS]
    assert all(len(x['U1_returns'])==5 and x['U1_decisions']==500 for x in chosen)
    assert all(x['step']==10000 and x['gpu']=='Tesla V100-PCIE-32GB' for x in chosen)
    for ti,(t,x) in enumerate(zip(TASKS,chosen)):
        expert,rand = REF[t]
        returns[ci,ti] = (np.asarray(x['U1_returns'])-rand)/(expert-rand)
        assert np.isclose(np.mean(x['U1_returns']), x['U1_return'])
    row = {'sampler_steps':s,'replan_decisions':r,'score':returns[ci].mean(),
           'planning_seconds_per_agent_decision':np.mean([x['t_u1_plan_s']/2500 for x in chosen]),
           'total_seconds_per_agent_decision':np.mean([x['t_total_s']/2500 for x in chosen]),
           'mean_batch_planning_call_seconds':np.mean([x['t_u1_plan_s']/np.ceil(500/r) for x in chosen]),
           'planning_hours_observed_5_episodes':np.mean([x['t_u1_plan_s']/3600 for x in chosen])}
    row.update(dict(zip(TASKS,returns[ci].mean(1))))
    rows.append(row)

# Shared within-task bootstrap indices preserve pairing across configurations.
# Tasks are fixed; no task or training-seed generalisation is estimated.
rng = np.random.default_rng(20261004)
B = 20000
indices = rng.integers(0,5,size=(3,B,5))
boot = np.stack([np.stack([returns[ci,ti][indices[ti]].mean(1) for ti in range(3)]).mean(0) for ci in range(6)])
contrasts = []
for a,b,label in [(5,0,'selected minus original default'),(5,1,'selected minus highest observed score')]:
    delta = returns[a]-returns[b]
    ci = np.quantile(boot[a]-boot[b],[.025,.975])
    loo = np.array([np.delete(delta,j,axis=1).mean() for j in range(5)])
    contrasts.append({'comparison':label,'difference':delta.mean(),'percentile_95_low':ci[0],
                      'percentile_95_high':ci[1],'leave_one_seed_out_min':loo.min(),
                      'leave_one_seed_out_max':loo.max(), 'per_task_differences':dict(zip(TASKS,delta.mean(1).tolist()))})

cost=np.array([x['planning_seconds_per_agent_decision'] for x in rows])
scores=returns.mean((1,2))
best=scores.max()
chosen=int(np.argmin(np.where(scores>=best-.05,cost,np.inf)))
assert chosen==5
out={'source_commit':raw['commit'],'source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
     'n_configs':6,'n_tasks':3,'episodes_per_cell':5,'n_episodes':90,
     'selected_config':CONFIGS[chosen],'speedup':cost[0]/cost[5],
     'cost_reduction_percent':100*(1-cost[5]/cost[0]),
     'total_observed_pilot_gpu_hours':sum(x['t_total_s'] for x in records)/3600,
     'rows':rows,'paired_contrasts':contrasts,
     'inference_note':'20,000 paired within-task percentile resamples; fixed checkpoint/tasks; no selection adjustment; not equivalence evidence.'}
(ROOT/'evidence'/'derived_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
with (ROOT/'evidence'/'configuration_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
fig,axs=plt.subplots(1,2,figsize=(10,3.7),gridspec_kw={'width_ratios':[1.1,1]})
colors={4:'#255F85',8:'#B05A28'}
for r in [4,8]:
    ids=[i for i,c in enumerate(CONFIGS) if c[1]==r]
    axs[0].plot(cost[ids],scores[ids],color=colors[r],marker='o',label=f'Replan every {r} decisions')
    for i in ids:
        offset={0:(-21,-17),1:(-7,8),2:(3,-18),3:(3,8),4:(3,8),5:(-13,8)}[i]
        axs[0].annotate(f'{CONFIGS[i][0]} steps',(cost[i],scores[i]),xytext=offset,textcoords='offset points',fontsize=8)
axs[0].scatter([cost[5]],[scores[5]],s=170,facecolors='none',edgecolors='black',linewidths=1.2,zorder=5)
axs[0].set(xlabel='Planning seconds / agent decision',ylabel='Mean normalised return',ylim=(.06,.29),xlim=(2.5,11.4),title='A. Observed cost and return')
axs[0].legend(loc='lower right',fontsize=8,frameon=False)
x=np.arange(6);width=.24
for ti,t in enumerate(TASKS):
    axs[1].bar(x+(ti-1)*width,returns[:,ti].mean(1),width=width,label=t.split('-')[0],color=['#255F85','#68A08A','#B05A28'][ti])
axs[1].axhline(0,color='#888888',linewidth=.7)
axs[1].set(xticks=x,xticklabels=[f'{s}/{r}' for s,r in CONFIGS],xlabel='Sampler steps / replanning interval',ylabel='Normalised return',title='B. Task means (five episodes each)')
axs[1].legend(fontsize=8,frameon=False,loc='upper left')
fig.tight_layout(w_pad=2)
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures'/'cost_return.png',bbox_inches='tight')
fig.savefig(ROOT/'figures'/'cost_return.svg',bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(9,2.6))
for i,(s,r) in enumerate(CONFIGS):
    vals=[next(x for x in records if x['task']=='cup-catch' and x['eval_settings']['steps']==s and x['eval_settings']['replan_blocks']*4==r)['U1_returns']][0]
    ax.scatter(np.arange(5)+1,[i]*5,s=60,c=vals,cmap='viridis',vmin=0,vmax=1000,edgecolor='black',linewidth=.4)
    for j,v in enumerate(vals):ax.annotate(str(int(v)),(j+1,i),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
ax.set(yticks=range(6),yticklabels=[f'{s} steps / {r} decisions' for s,r in CONFIGS],xticks=range(1,6),xlabel='Paired episode index',ylim=(-.5,5.7),title='Cup-catch: individual returns behind the averages')
fig.tight_layout();fig.savefig(ROOT/'figures'/'cup_episodes.png',bbox_inches='tight');plt.close(fig)
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

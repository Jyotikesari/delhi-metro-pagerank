import pickle, json, numpy as np, networkx as nx, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
s=pickle.load(open('out/state.pkl','rb')); G=s['G']; nodes=s['nodes']; pr=s['pr']; deg=s['deg']; B=s['B']; imp=s['imp']
r=json.load(open('out/results.json'))
LC={'red':'#d62728','yellow':'#e6b800','blue':'#1f77b4','green':'#2ca02c','violet':'#7b3fa0','magenta':'#d6249f','pink':'#f08fb0','orange':'#ff7f0e','aqua':'#17b8be','gray':'#7f7f7f','rapid metro':'#555555'}
pretty=lambda n:{'durgabai deshmukh south campus':'Dhaula Kuan / D. Deshmukh S.Campus'}.get(n,n.title())
# ---- Fig 1: schematic network, size ~ PageRank
pos=nx.kamada_kawai_layout(G)
fig,ax=plt.subplots(figsize=(6.6,3.9))
ecol=[]
line_of_edge={}
import build_graph
segs,_,_=build_graph.load()
for line,st in segs:
    for a,b in zip(st,st[1:]):
        if a!=b: line_of_edge[frozenset((a,b))]=line
for a,b in G.edges():
    ax.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],color=LC.get(line_of_edge.get(frozenset((a,b)),'gray'),'#999'),lw=1.8,zorder=1,alpha=.9)
sz=((pr-pr.min())/(pr.max()-pr.min()))**1.5*150+3
ax.scatter([pos[v][0] for v in nodes],[pos[v][1] for v in nodes],s=sz,c=['k' if deg[i]>=3 else 'white' for i in range(len(nodes))],edgecolors='k',linewidths=.5,zorder=2)
off={'kashmere gate':(34,-34),'sikandarpur':(-10,14),'azadpur':(-70,22),'kalkaji mandir':(10,-22),'welcome':(18,14)}
for i in np.argsort(-pr)[:5]:
    v=nodes[i]; ax.annotate(f"{i and ''}{pretty(v)}",pos[v],xytext=off.get(v,(6,6)),textcoords='offset points',fontsize=7,weight='bold',zorder=3,
        arrowprops=dict(arrowstyle='-',lw=.4))
ax.set_axis_off()
from matplotlib.lines import Line2D
h=[Line2D([0],[0],color=c,lw=2,label=k.title()) for k,c in LC.items()]
ax.legend(handles=h,ncol=1,fontsize=6.5,frameon=False,loc='center left',bbox_to_anchor=(1.0,.5),title='Line',title_fontsize=7)
plt.savefig('out/fig1_network.png',dpi=220,bbox_inches='tight',pad_inches=0.03); plt.close()
# ---- Fig 2: convergence + damping
sw=r['sweep']
fig,ax=plt.subplots(1,3,figsize=(6.9,2.15))
x0=np.zeros(len(nodes)); x0[nodes.index('kashmere gate')]=1
Pm=nx.to_numpy_array(G,nodelist=nodes); Pm=Pm/Pm.sum(1)[:,None]
y=x0.copy(); ed=[]
for k in range(300):
    y=0.85*(y@Pm)+0.15/len(nodes); ed.append(np.abs(y-pr).sum())
ed=np.maximum(ed,1e-16); kd=int(np.argmax(ed<1e-6))+1
ax[0].semilogy(range(1,301),ed,color='C0',label='damped, d=0.85')
P=nx.to_numpy_array(G,nodelist=nodes); P=P/P.sum(1)[:,None]; stat=deg/deg.sum()
x=np.zeros(len(nodes)); x[nodes.index('kashmere gate')]=1; e=[]
for k in range(6000):
    x=x@P; e.append(np.abs(x-stat).sum())
ax[0].semilogy(range(1,6001),e,color='C3',label='undamped walk (d=1)')
ax[0].axhline(1e-6,color='gray',ls=':',lw=.8)
ku=int(np.argmax(np.array(e)<1e-6))+1 if (np.array(e)<1e-6).any() else None
ax[0].set_xscale('log'); ax[0].set_xlabel('iteration k'); ax[0].set_ylabel('L1 distance to fixed point'); ax[0].legend(fontsize=6,frameon=False); ax[0].set_title('(a) Convergence',fontsize=8)
d=[t['d'] for t in sw]
ax[1].plot(d,[t['iters'] for t in sw],'o-',ms=3); ax[1].set_yscale('log'); ax[1].set_xlabel('damping d'); ax[1].set_ylabel('iterations (Δ<1e-10)'); ax[1].set_title('(b) Cost of damping',fontsize=8)
ax[2].plot(d,[t['tau'] for t in sw],'s-',ms=3,color='C2',label='Kendall τ vs d=0.85')
ax[2].plot(d,[t['overlap10']/10 for t in sw],'^-',ms=3,color='C1',label='top-10 overlap')
ax[2].set_ylim(.75,1.02); ax[2].set_xlabel('damping d'); ax[2].legend(fontsize=6,frameon=False,loc='lower center'); ax[2].set_title('(c) Ranking stability',fontsize=8)
plt.tight_layout(); plt.savefig('out/fig2_damping.png',dpi=220); plt.close()
print('undamped iterations to 1e-6:',ku,'damped:',kd)
json.dump({'damped_iters_1e-6':kd,'undamped_iters_1e-6':ku,'undamped_err_6000':e[-1]},open('out/extra.json','w'))
# ---- Fig 3: robustness
fig,ax=plt.subplots(1,2,figsize=(6.9,2.4))
cols={'PageRank':'C0','Degree':'C1','Betweenness':'C3','Random (mean of 100)':'gray'}
for k,v in r['curves'].items():
    ax[0].plot(range(len(v['S'])),v['S'],label=k,color=cols[k],lw=1.4,ls='--' if 'Random' in k else '-')
    ax[1].plot(range(len(v['E'])),v['E'],label=k,color=cols[k],lw=1.4,ls='--' if 'Random' in k else '-')
ax[0].set_xlabel('stations removed'); ax[0].set_ylabel('giant component / n'); ax[0].set_title('(a) Connectivity',fontsize=8)
ax[1].set_xlabel('stations removed'); ax[1].set_ylabel('global efficiency / E₀'); ax[1].set_title('(b) Efficiency',fontsize=8)
ax[0].legend(fontsize=6,frameon=False)
plt.tight_layout(); plt.savefig('out/fig3_robust.png',dpi=220); plt.close()
# ---- Fig 4: PR vs impact scatter (for poster)
from scipy.stats import spearmanr
fig,ax=plt.subplots(1,2,figsize=(6.2,2.4),sharey=True)
ax[0].scatter(pr,imp*100,s=7,alpha=.6,color='C0'); ax[0].set_xlabel('PageRank'); ax[0].set_ylabel('efficiency lost if closed (%)')
ax[0].set_title(f"ρ = {r['single_corr']['pr']:.2f}",fontsize=8)
ax[1].scatter(B,imp*100,s=7,alpha=.6,color='C3'); ax[1].set_xlabel('Betweenness'); ax[1].set_title(f"ρ = {r['single_corr']['bet']:.2f}",fontsize=8)
plt.tight_layout(); plt.savefig('out/fig4_scatter.png',dpi=220); plt.close()

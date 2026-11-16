import pickle, json, numpy as np, networkx as nx, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import build_graph
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':15,'legend.fontsize':13})
s=pickle.load(open('out/state.pkl','rb')); G=s['G']; nodes=s['nodes']; pr=s['pr']; deg=s['deg']; B=s['B']; imp=s['imp']
r=json.load(open('out/results.json'))
LC={'red':'#d62728','yellow':'#e6b800','blue':'#1f77b4','green':'#2ca02c','violet':'#7b3fa0','magenta':'#d6249f','pink':'#f08fb0','orange':'#ff7f0e','aqua':'#17b8be','gray':'#7f7f7f','rapid metro':'#555555'}
pretty=lambda n:{'durgabai deshmukh south campus':'Dhaula Kuan'}.get(n,n.title())
pos=nx.kamada_kawai_layout(G)
segs,_,_=build_graph.load(); loe={}
for line,st in segs:
    for a,b in zip(st,st[1:]):
        if a!=b: loe[frozenset((a,b))]=line
fig,ax=plt.subplots(figsize=(9,5.4))
for a,b in G.edges():
    ax.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],color=LC.get(loe.get(frozenset((a,b)),'gray'),'#999'),lw=3,zorder=1,alpha=.9)
sz=((pr-pr.min())/(pr.max()-pr.min()))**1.5*330+6
ax.scatter([pos[v][0] for v in nodes],[pos[v][1] for v in nodes],s=sz,c=['k' if deg[i]>=3 else 'white' for i in range(len(nodes))],edgecolors='k',linewidths=.8,zorder=2)
off={'kashmere gate':(40,-45),'sikandarpur':(-20,22),'azadpur':(-110,34),'kalkaji mandir':(14,-36),'welcome':(30,24)}
for i in np.argsort(-pr)[:5]:
    v=nodes[i]; ax.annotate(pretty(v),pos[v],xytext=off.get(v,(6,6)),textcoords='offset points',fontsize=15,weight='bold',zorder=3,arrowprops=dict(arrowstyle='-',lw=.8))
ax.set_axis_off()
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],color=c,lw=4,label=k.title()) for k,c in LC.items()],fontsize=11.5,frameon=False,loc='center left',bbox_to_anchor=(0.99,.5))
plt.savefig('out/p_fig1.png',dpi=200,bbox_inches='tight',pad_inches=.05); plt.close()
# damping: 2 panels
P=nx.to_numpy_array(G,nodelist=nodes); P=P/P.sum(1)[:,None]; stat=deg/deg.sum()
x=np.zeros(len(nodes)); x[nodes.index('kashmere gate')]=1; e=[]
for k in range(6000):
    x=x@P; e.append(np.abs(x-stat).sum())
y=np.zeros(len(nodes)); y[nodes.index('kashmere gate')]=1; ed=[]
for k in range(300):
    y=0.85*(y@P)+0.15/len(nodes); ed.append(np.abs(y-pr).sum())
ed=np.maximum(ed,1e-16); sw=r['sweep']; d=[t['d'] for t in sw]
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
ax[0].semilogy(range(1,301),ed,color='C0',lw=3,label='damped (d = 0.85)'); ax[0].semilogy(range(1,6001),e,color='C3',lw=3,label='undamped (d = 1)')
ax[0].axhline(1e-6,color='gray',ls=':'); ax[0].set_xscale('log'); ax[0].set_xlabel('iteration'); ax[0].set_ylabel('error (L1)'); ax[0].legend(frameon=False,loc='lower left'); ax[0].set_title('Convergence speed',fontsize=16,weight='bold')
ax[1].plot(d,[t['tau'] for t in sw],'s-',lw=3,ms=8,color='C2',label='rank similarity (Kendall τ)')
ax[1].plot(d,[t['overlap10']/10 for t in sw],'^-',lw=3,ms=8,color='C1',label='top-10 overlap')
ax[1].set_ylim(.78,1.03); ax[1].set_xlabel('damping factor d'); ax[1].legend(frameon=False,loc='lower center',fontsize=12); ax[1].set_title('Ranking stability',fontsize=16,weight='bold')
plt.tight_layout(); plt.savefig('out/p_fig2.png',dpi=200); plt.close()
# robustness
cols={'PageRank':'C0','Degree':'C1','Betweenness':'C3','Random (mean of 100)':'gray'}
fig,ax=plt.subplots(1,2,figsize=(9,3.8))
for k,v in r['curves'].items():
    ax[0].plot(range(len(v['S'])),v['S'],label=k.replace(' (mean of 100)',''),color=cols[k],lw=3,ls='--' if 'Random' in k else '-')
    ax[1].plot(range(len(v['E'])),v['E'],color=cols[k],lw=3,ls='--' if 'Random' in k else '-')
ax[0].set_xlabel('stations removed'); ax[0].set_ylabel('giant component'); ax[0].set_title('Connectivity',fontsize=16,weight='bold'); ax[0].legend(frameon=False)
ax[1].set_xlabel('stations removed'); ax[1].set_ylabel('efficiency (rel.)'); ax[1].set_title('Efficiency',fontsize=16,weight='bold')
plt.tight_layout(); plt.savefig('out/p_fig3.png',dpi=200); plt.close()
fig,ax=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
ax[0].scatter(pr*1000,imp*100,s=28,alpha=.6,color='C0'); ax[0].set_xlabel('PageRank (×10⁻³)'); ax[0].set_ylabel('efficiency lost if closed (%)'); ax[0].set_title(f"PageRank:  ρ = {r['single_corr']['pr']:.2f}",fontsize=16,weight='bold')
ax[1].scatter(B,imp*100,s=28,alpha=.6,color='C3'); ax[1].set_xlabel('Betweenness'); ax[1].set_title(f"Betweenness:  ρ = {r['single_corr']['bet']:.2f}",fontsize=16,weight='bold')
plt.tight_layout(); plt.savefig('out/p_fig4.png',dpi=200); plt.close()
print('ok')

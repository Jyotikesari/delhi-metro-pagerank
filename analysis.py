import numpy as np, networkx as nx, json, random
from scipy.stats import spearmanr, kendalltau
from build_graph import load
np.random.seed(7); random.seed(7)
segs,conn,df=load()
G=nx.Graph(); lines={}
for line,st in segs:
    for a in st: lines.setdefault(a,set()).add(line)
    for a,b in zip(st,st[1:]):
        if a!=b: G.add_edge(a,b)
nodes=sorted(G.nodes()); idx={v:i for i,v in enumerate(nodes)}; n=len(nodes)
A=nx.to_numpy_array(G,nodelist=nodes); deg=A.sum(1); P=A/deg[:,None]   # row-stochastic
def pagerank_power(P,d=0.85,tol=1e-10,maxit=10000):
    n=len(P); x=np.ones(n)/n; hist=[]
    for k in range(1,maxit+1):
        xn=d*(x@P)+(1-d)/n
        err=np.abs(xn-x).sum(); hist.append(err); x=xn
        if err<tol: break
    return x,k,hist
def pagerank_closed(A,d=0.85):
    n=len(A); P=A/A.sum(1)[:,None]
    return (1-d)/n*np.linalg.solve(np.eye(n)-d*P.T,np.ones(n))
pr,it,hist=pagerank_power(P)
pr_c=pagerank_closed(A); pr_nx=np.array([nx.pagerank(G,alpha=.85,tol=1e-12,max_iter=2000)[v] for v in nodes])
res={'n':n,'m':G.number_of_edges(),'iters_085':it,
     'max_diff_power_vs_closed':float(np.abs(pr-pr_c).max()),'max_diff_vs_networkx':float(np.abs(pr-pr_nx).max()),
     'avg_deg':float(deg.mean()),'max_deg':int(deg.max()),'diameter':nx.diameter(G),'avg_path':nx.average_shortest_path_length(G),
     'clustering':nx.average_clustering(G),
     'n_interchange':int(sum(1 for v in nodes if len(lines[v])>1)),
     'sum_pr':float(pr.sum())}
# degree-proportional stationary dist (d=1)
stat=deg/deg.sum()
res['corr_pr_stationary_pearson']=float(np.corrcoef(pr,stat)[0,1])
bet=nx.betweenness_centrality(G); clo=nx.closeness_centrality(G)
B=np.array([bet[v] for v in nodes]); C=np.array([clo[v] for v in nodes])
eig=nx.eigenvector_centrality_numpy(G); Ev=np.array([eig[v] for v in nodes])
def rho(a,b): return float(spearmanr(a,b)[0])
res['spearman']={'deg':rho(pr,deg),'betweenness':rho(pr,B),'closeness':rho(pr,C),'eigenvector':rho(pr,Ev)}
res['spearman_deg_bet']=rho(deg,B)
order=np.argsort(-pr)
top=[]
for r,i in enumerate(order[:10],1):
    top.append(dict(rank=r,station=nodes[i],pr=float(pr[i]),deg=int(deg[i]),bet=float(B[i]),
                    bet_rank=int((B>B[i]).sum()+1),lines=sorted(lines[nodes[i]])))
res['top10']=top
# PR / degree residual: who gains/loses relative to d=1 limit
ratio=pr/stat
res['ratio_range']=[float(ratio.min()),float(ratio.max())]
gain=np.argsort(-ratio)[:5]; lose=np.argsort(ratio)[:5]
res['gain']=[(nodes[i],float(ratio[i]),int(deg[i])) for i in gain]
res['lose']=[(nodes[i],float(ratio[i]),int(deg[i])) for i in lose]
# damping sweep
ds=[0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99]; sweep=[]
base_top10=set(np.argsort(-pr)[:10])
for d in ds:
    x,k,_=pagerank_power(P,d)
    sweep.append(dict(d=d,iters=k,tau=float(kendalltau(x,pr)[0]),overlap10=len(base_top10&set(np.argsort(-x)[:10])),
                      tau_deg=float(kendalltau(x,deg)[0])))
res['sweep']=sweep
# second eigenvalue of P (convergence rate)
w=np.sort(np.abs(np.linalg.eigvals(P)))[::-1]; res['lambda2']=float(w[1]); res['lambda_min']=float(np.sort(np.linalg.eigvals(P).real)[0])
# robustness
def eff(H):
    N=n  # normalise by original size so removed nodes count as lost
    tot=0.0
    for s,dd in nx.all_pairs_shortest_path_length(H):
        tot+=sum(1.0/v for v in dd.values() if v>0)
    return tot/(N*(N-1))
def giant(H):
    return max((len(c) for c in nx.connected_components(H)),default=0)/n
def pr_of(H):
    return nx.pagerank(H,alpha=.85,tol=1e-10,max_iter=2000)
strat={'PageRank':lambda H:max(pr_of(H).items(),key=lambda kv:kv[1])[0],
       'Degree':lambda H:max(dict(H.degree()).items(),key=lambda kv:kv[1])[0],
       'Betweenness':lambda H:max(nx.betweenness_centrality(H).items(),key=lambda kv:kv[1])[0]}
K=30; curves={}
E0=eff(G); res['E0']=E0
for name,f in strat.items():
    H=G.copy(); S=[1.0]; Es=[1.0]; removed=[]
    for k in range(K):
        v=f(H); H.remove_node(v); removed.append(v)
        S.append(giant(H)); Es.append(eff(H)/E0)
    curves[name]=dict(S=S,E=Es,removed=removed)
rS=np.zeros(K+1); rE=np.zeros(K+1); T=100
for t in range(T):
    H=G.copy(); order_r=random.sample(nodes,K); S=[1.0];Es=[1.0]
    for v in order_r:
        H.remove_node(v); S.append(giant(H)); Es.append(eff(H)/E0)
    rS+=np.array(S); rE+=np.array(Es)
curves['Random (mean of 100)']=dict(S=list(rS/T),E=list(rE/T))
res['curves']=curves
# single-station removal impact
imp=np.zeros(n); impS=np.zeros(n)
for v in nodes:
    H=G.copy(); H.remove_node(v); imp[idx[v]]=1-eff(H)/E0; impS[idx[v]]=1-giant(H)*n/(n-1) if False else giant(H)
res['single_corr']={'pr':rho(pr,imp),'deg':rho(deg,imp),'bet':rho(B,imp)}
oi=np.argsort(-imp)[:10]
res['single_top']=[(nodes[i],float(imp[i]),int(np.where(order==i)[0][0]+1),int((B>B[i]).sum()+1)) for i in oi]
# top-10 overlap of single-removal impact with each measure
t10=set(np.argsort(-imp)[:10])
res['overlap_top10_impact']={'pr':len(t10&set(order[:10])),'deg':len(t10&set(np.argsort(-deg)[:10])),'bet':len(t10&set(np.argsort(-B)[:10]))}
# articulation points
ap=list(nx.articulation_points(G)); res['n_articulation']=len(ap)
res['frac_articulation']=len(ap)/n
json.dump(res,open('out/results.json','w'),indent=1,default=str)
np.save('out/pr.npy',pr)
import pickle; pickle.dump(dict(G=G,nodes=nodes,pr=pr,deg=deg,B=B,lines=lines,hist=hist,imp=imp),open('out/state.pkl','wb'))
print(json.dumps({k:v for k,v in res.items() if k not in('curves',)},indent=1,default=str))
for k,v in res['curves'].items(): print(k,'S@10,30',round(v['S'][10],3),round(v['S'][30],3),'E@10,30',round(v['E'][10],3),round(v['E'][30],3))

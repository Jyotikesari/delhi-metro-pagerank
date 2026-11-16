import pandas as pd, re, networkx as nx, json
# aliases = same physical interchange listed under two names; each pair is confirmed by the dataset's own 'Conn:' rows
ALIAS={'new delhi-airport express':'new delhi','dhaula kuan':'durgabai deshmukh south campus','noida sector 51':'noida sector 52'}
def norm(s):
    s=str(s)
    s=re.sub(r'\(.*?\)','',s)
    s=re.sub(r'conn:.*','',s,flags=re.I)
    s=re.sub(r'\bmetro station\b','',s,flags=re.I)
    s=re.sub(r'[^a-z0-9\- ]',' ',s.lower())
    s=re.sub(r'\s+',' ',s).strip()
    s=s.replace('subash','subhash').replace('extention','extension')
    return ALIAS.get(s,s)
def load():
    df=pd.read_csv('a/Data/raw_Delhi_metro.csv')
    df.columns=['id','name','dist','line','opened','layout','lat','lon']
    segs=[];conn={};last=None;cur=None
    for _,r in df.iterrows():
        nm=str(r['name'])
        if nm.startswith('Conn:'):
            if last: conn.setdefault(last,set()).add(nm.split(':')[1].strip().lower())
            continue
        if pd.isna(r.id): continue
        line=re.sub(r' line( branch)?$','',r.line.strip().lower()).replace('voilet','violet')
        if cur is None or line!=cur[0]:
            cur=[line,[]];segs.append(cur)
        n=norm(nm);cur[1].append(n);last=n
        if 'Conn:' in nm: conn.setdefault(n,set()).add(re.search(r'Conn:\s*(\w+)',nm).group(1).lower())
    return segs,conn,df
if __name__=='__main__':
    segs,conn,df=load()
    G=nx.Graph();lines={}
    for line,st in segs:
        for a in st: lines.setdefault(a,set()).add(line.replace(' branch',''))
        for a,b in zip(st,st[1:]): G.add_edge(a,b)
    print(G.number_of_nodes(),G.number_of_edges(),nx.number_connected_components(G))
    print([len(c) for c in nx.connected_components(G)])
    # compare conn annotations
    for s,c in sorted(conn.items()):
        ls=lines.get(s,set())
        exp=len(c)+1
        flag='' if len(ls)>=exp else '  <-- MISMATCH'
        print(f"{s:28s} conn={sorted(c)} merged_lines={sorted(ls)}{flag}")

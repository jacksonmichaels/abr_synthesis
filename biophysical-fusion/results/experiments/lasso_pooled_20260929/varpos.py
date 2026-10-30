import numpy as np
z=np.load("/home/jacksonmicha_umass_edu/abr_workspace/wholegenome_cache.npz",allow_pickle=False)
keys=z["keys"]; n_iso=z["indptr"].shape[0]-1
G=4411532
pos=np.array([int(k.split(":",1)[0]) for k in keys])
cnt=np.bincount(z["indices"],minlength=len(keys))
up=np.unique(pos); print("isolates",n_iso,"keys",len(keys))
print("variable positions (>=1 key starts there):",len(up), f"{len(up)/G:.2%} of genome")
pc={}
for p,c in zip(pos,cnt): pc[p]=pc.get(p,0)+c
carriers=np.array(list(pc.values()))
print("positions where only 1 isolate varies:",(carriers==1).sum(), f"{(carriers==1).mean():.1%}")
print("positions with >=10 carriers:",(carriers>=10).sum())
nalle=np.bincount(np.unique(pos,return_inverse=True)[1])
print("positions with >=2 distinct alleles:",(nalle>=2).sum())
print("1s per isolate, delta:",round(z["indices"].shape[0]/n_iso))
print("1s per isolate, one-hot on variable positions only: ~",len(up))
print(f"total 1s one-hot-on-variable: {n_iso*len(up):.2e} vs delta {z['indices'].shape[0]:.2e}")

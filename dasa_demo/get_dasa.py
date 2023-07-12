import sys

fn=[]
for rl in open(sys.argv[1],'r'):
    fn.append(rl.split()[:][0])

_chain=[]
for rl in open(fn[0],'r'):
    atom=rl[13:15]
    chain=rl[21:22]
    if len(rl) >= 40 and atom=="CA":
        _chain.append(chain)

_asa_c=[]
_resid=[]
for rl in open("%s.xvg"%fn[0],'r'):
    srl=rl.split()[:]
    _asa_c.append(srl[1])
#    _resid.append(srl[0])

for rl in open(sys.argv[2],'r'):
    srl = rl.split()[:]
    if rl[:4] == "ATOM" and rl[13:15] == "CA":
        _resid.append(rl[22:27])



_asa_s=[]
for i in range(1,len(fn)):
    fname=fn[i]
    for rl in open("%s.xvg"%fname,'r'):
        srl=rl.split()[:]
        _asa_s.append(srl[1])

print("# ResidueID ChainID ASA_complex ASA_single dASA Interface")
for i in range(len(_chain)):
    RES=_resid[i]
    CHN=_chain[i]
    ASA_c=float(_asa_c[i])
    ASA_s=float(_asa_s[i])
    dASA=ASA_s-ASA_c
    if dASA>0.01:
        inter=1
    else:
        inter=0
    print("%6s %1s %6.3f %6.3f %6.3f %s"%(RES,CHN,ASA_c,ASA_s,dASA,inter))

"""Run every study-specific manuscript reproduction check.

Each child script reads only the archived input for its study and writes a
study-level checks.csv file. This runner stops with a non-zero exit status if
any study fails and otherwise writes a combined verification summary.
"""
from pathlib import Path
import subprocess, sys, pandas as pd
ROOT=Path(__file__).resolve().parent
jobs=[
("Hubner",ROOT/"studies/01_hubner/scripts/reproduce_hubner.py"),
("TBE",ROOT/"studies/02_tbe_exp3/scripts/reproduce_tbe_exp3.py"),
("Evans",ROOT/"studies/03_evans/scripts/reproduce_evans.py"),
("Bogacz",ROOT/"studies/04_bogacz/scripts/reproduce_bogacz.py"),
("Leng",ROOT/"studies/05_leng/scripts/reproduce_leng.py")]
ok=True
for name,p in jobs:
 print("\n"+"="*72+f"\n{name}\n"+"="*72)
 r=subprocess.run([sys.executable,str(p)],cwd=p.parents[1])
 ok &= r.returncode==0
if not ok: raise SystemExit(1)
rows=[]
for name,p in jobs:
 c=pd.read_csv(p.parents[1]/"results/checks.csv")
 rows.append({"study":name,"passed":int(c.PASS.sum()),"checks":len(c),"status":"PASS" if c.PASS.all() else "FAIL"})
out=pd.DataFrame(rows); (ROOT/"verification").mkdir(exist_ok=True)
out.to_csv(ROOT/"verification/summary.csv",index=False)
print("\n"+out.to_string(index=False))
print(f"\nTOTAL: {out.passed.sum()}/{out.checks.sum()} PASS")

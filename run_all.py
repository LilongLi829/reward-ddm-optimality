"""Run all manuscript reproduction scripts and rebuild the combined result ledger."""
from pathlib import Path
import subprocess, sys, pandas as pd
ROOT=Path(__file__).resolve().parent
studies=[
("Hübner","studies/01_hubner/scripts/reproduce_hubner.py"),
("TBE Exp3","studies/02_tbe_exp3/scripts/reproduce_tbe_exp3.py"),
("Evans","studies/03_evans/scripts/reproduce_evans.py"),
("Bogacz","studies/04_bogacz/scripts/reproduce_bogacz.py"),
("Leng","studies/05_leng/scripts/reproduce_leng.py")]
figures=[
"studies/01_hubner/figures/Figure1_Hubner_SATF.png",
"studies/01_hubner/figures/Figure2_Hubner_LOPO.png",
"studies/02_tbe_exp3/figures/Figure3_TBE_temporal.png",
"studies/03_evans/figures/Figure4_Evans_specificity.png",
"studies/04_bogacz/figures/Figure5_Bogacz_boundary.png",
"studies/05_leng/figures/Figure6_Leng_effects.png",
"studies/05_leng/figures/Figure7_Leng_mapping.png"]
summary=[]; ledger=[]; ok=True
for study,rel in studies:
    p=subprocess.run([sys.executable,str(ROOT/rel)],cwd=ROOT)
    checks=ROOT/Path(rel).parents[1]/"results/checks.csv"
    d=pd.read_csv(checks)
    n=int(d.PASS.astype(bool).sum()); total=len(d); passed=(p.returncode==0 and n==total)
    summary.append({"study":study,"passed":n,"total":total,"status":"PASS" if passed else "FAIL"})
    ok &= passed
    for _,r in d.iterrows():
        ledger.append({"study":study,"metric":r.metric,"expected":r.expected,"recomputed":r.recomputed,
                       "tolerance":r.tolerance,"status":"PASS" if bool(r.PASS) else "FAIL",
                       "generator":rel,"input_stage":"archived derived/frozen analysis input"})
for rel in figures:
    exists=(ROOT/rel).exists()
    summary.append({"study":"Figure regeneration","passed":1 if exists else 0,"total":1,"status":"PASS" if exists else "FAIL"})
    ok &= exists
(ROOT/"verification").mkdir(exist_ok=True)
pd.DataFrame(summary).to_csv(ROOT/"verification/summary.csv",index=False)
(ROOT/"manuscript").mkdir(exist_ok=True)
pd.DataFrame(ledger).to_csv(ROOT/"manuscript/MANUSCRIPT_RESULT_LEDGER.csv",index=False)
print(pd.DataFrame(summary).to_string(index=False))
print(f"\nNumerical checks: {sum(x['passed'] for x in summary if x['study']!='Figure regeneration')}/"
      f"{sum(x['total'] for x in summary if x['study']!='Figure regeneration')} PASS")
print("Figures: 7/7 regenerated" if all((ROOT/f).exists() for f in figures) else "Figures: regeneration failure")
raise SystemExit(0 if ok else 1)

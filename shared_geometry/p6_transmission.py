"""P6-1: action-common vs action-specific errors in predicted visible geometry.

Read-only matched P1/P4/P5 analysis. Seven *gripper-stencil* positions at a
given index are not the same world point across actions. Clean is privileged.
Continuous decomposition uses a strict triple-source, all-seven-observed mask;
unknown rays never contribute artificial zero error.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .data import load_signed,sha256,signed,atomic_json,frame_key
from .p1_clean_visible import read_cache_index,load_cached
from .p3_action_geometry_diagnostics import REGIONS
from .p0_relative_decision import scene_balanced,scene_bootstrap
from .p4_common import new_directory,write_records
from .p6_opportunity import load_matched,identity

SCHEMA="shared-visible-geometry-p6-transmission-v1"
CHANNELS=("udf","projective")
REGION_MASKS={"all":np.ones(75,bool),**REGIONS}
METRICS=("pred_total_mse_mm2","corr_total_mse_mm2","pred_shared_mse_mm2",
         "corr_shared_mse_mm2","pred_contrast_mse_mm2","corr_contrast_mse_mm2",
         "total_improvement_mm2","shared_improvement_mm2",
         "contrast_improvement_mm2","action_relative_mae_improvement_mm",
         "correction_common_energy_fraction","clean_delta_magnitude_mm")


def decompose(pred,corrected,clean,tau_m):
    """Orthogonal e=mean_actions(e)+centered_actions(e) MSE decomposition."""
    p,c,t=[np.asarray(v,dtype=np.float64) for v in (pred,corrected,clean)]
    if any(v.shape!=(7,75,6) or not np.isfinite(v).all() for v in (p,c,t)):
        raise ValueError("Expected finite matched [7,75,6] geometry")
    if not 0<tau_m<=.2:raise ValueError("Invalid truncation")
    valid=np.logical_and.reduce([v[...,2:].argmax(-1)!=3 for v in (p,c,t)])
    all_known=valid.all(axis=0)
    out=dict(valid_stencil_sites=int(all_known.sum()),
             all_action_known_fraction=float(all_known.mean()),
             action_site_known_fraction=float(valid.mean()))
    mm=tau_m*1000.
    for region,sites in REGION_MASKS.items():
        selected=all_known&sites
        out[region+"_n_sites"]=int(selected.sum())
        for ch,label in enumerate(CHANNELS):
            prefix=region+"_"+label+"_"
            if not selected.any():
                out.update({prefix+k:None for k in METRICS})
                continue
            ep=(p[:,selected,ch]-t[:,selected,ch])*mm
            ec=(c[:,selected,ch]-t[:,selected,ch])*mm
            def parts(e):
                mean=e.mean(axis=0,keepdims=True);center=e-mean
                total=float(np.mean(e**2));common=float(np.mean(mean**2))
                contrast=float(np.mean(center**2))
                if abs(total-common-contrast)>1e-8:
                    raise AssertionError("MSE decomposition failed")
                return total,common,contrast
            pt,ps,pc=parts(ep);ct,cs,cc=parts(ec)
            change=(c[:,selected,ch]-p[:,selected,ch])*mm
            total_energy=float(np.mean(change**2))
            common_energy=float(np.mean(change.mean(axis=0)**2))
            truth=(t[1:,selected,ch]-t[:1,selected,ch])*mm
            dp=(p[1:,selected,ch]-p[:1,selected,ch])*mm
            dc=(c[1:,selected,ch]-c[:1,selected,ch])*mm
            vals=(pt,ct,ps,cs,pc,cc,pt-ct,ps-cs,pc-cc,
                  float(np.abs(dp-truth).mean()-np.abs(dc-truth).mean()),
                  common_energy/total_energy if total_energy else None,
                  float(np.abs(truth).mean()))
            out.update({prefix+k:v for k,v in zip(METRICS,vals)})
    return out


def analyze(p1_index,p4_summary,p5_summary,output,bootstrap=5000,seed=0,smoke=False):
    if bootstrap<1:raise ValueError("Positive bootstrap budget required")
    pi=read_cache_index(p1_index,"validation",smoke)
    p4=load_signed(p4_summary);p5,modes,keys=load_matched(p5_summary,smoke)
    if (p4.get("study")!="P4-1 ordinary-depth, frozen P2 readout" or
        pi["fingerprint"]!=p4["sources"]["p1_index"] or
        pi["fingerprint"]!=p5["sources"]["p1_index_fingerprint"] or
        p4["sources"]["reader_sha256"]!=p5["sources"]["p2_reader_sha256"] or
        p4["sources"]["corrector_sha256"]!=p5["sources"]["depth_weight_sha256"]):
        raise ValueError("P1/P4/P5 frozen provenance mismatch")
    p1={frame_key(x["row"]):x for x in pi["entries"]}
    frames=p4["artifacts"]["frames"]
    if (len(frames)!=len(p1) or
        {frame_key(x["row"]) for x in frames}!=set(p1) or
        (not smoke and (len(frames)!=160 or len(keys)!=5120))):
        raise ValueError("Incomplete P4/P1 frame coverage")
    result=[]
    for f in frames:
        key=frame_key(f["row"]);old=p1[key]
        if sha256(f["path"])!=f["sha256"] or f["p1_frame_sha256"]!=old["sha256"]:
            raise ValueError("P4 physical artifact signature mismatch")
        cache=load_cached(old)
        with np.load(f["path"],allow_pickle=False) as z:
            cp=np.asarray(z["corrected_physics"],np.float32)
            ids=np.asarray(z["physical_id"]).astype(str)
            families=np.asarray(z["family_id"])
        pp=cache["predicted_physics"];cl=cache["clean_visible_physics"]
        if pp.shape!=cp.shape or pp.shape!=cl.shape or not np.array_equal(
            ids,np.asarray(cache["physical_id"]).astype(str)) or not np.array_equal(
            families,cache["family_id"]):
            raise ValueError("P4/P1 token or physical action mismatch")
        for n,family in enumerate(families):
            ik=(int(f["row"]["scene"]),f["row"]["domain"],
                int(f["row"]["anno"]),int(family))
            if ik not in modes["predicted_p2_fixed"]:raise ValueError("P5 family missing")
            a=modes["predicted_p2_fixed"][ik];b=modes["corrected_p2_fixed"][ik]
            if (a["physical_ids"]!=ids[n].tolist() or b["physical_ids"]!=ids[n].tolist() or
                a["native_count"]!=int(cache["quality_count"][n,0])):
                raise ValueError("P5 physical/quality identity mismatch")
            r=decompose(pp[n],cp[n],cl[n],pi["contract"]["tau"])
            r.update(scene=ik[0],domain=ik[1],anno=ik[2],family=ik[3],
                     native_count=a["native_count"],opportunity=a["opportunity"],
                     predicted_gain=a["gain"],corrected_gain=b["gain"],
                     delta_gain=b["gain"]-a["gain"],
                     improved_decision=bool(b["gain"]>a["gain"]),
                     harmed_decision=bool(b["gain"]<a["gain"]))
            result.append(r)
    if len(result)!=len(keys) or {identity(r) for r in result}!=set(keys):
        raise ValueError("P6-1 action family coverage mismatch")
    expected=p5["scores"]["corrected_p2_fixed"]["gain"]-p5["scores"]["predicted_p2_fixed"]["gain"]
    if abs(scene_balanced(result,"delta_gain")-expected)>2e-6:
        raise ValueError("P6-1 fixed-reader P5 gain parity failed")
    subsets={"all":result,"has_opportunity":[r for r in result if r["opportunity"]],
             "no_opportunity":[r for r in result if not r["opportunity"]],
             "decision_improved":[r for r in result if r["improved_decision"]],
             "decision_harmed":[r for r in result if r["harmed_decision"]]}
    groups={}
    for name,rows in subsets.items():
        fields={"n_families":len(rows)}
        for region in REGION_MASKS:
            for channel in CHANNELS:
                for metric in METRICS:
                    k=region+"_"+channel+"_"+metric
                    fields[k]=scene_balanced(rows,k) if rows else None
        fields["delta_gain"]=scene_balanced(rows,"delta_gain") if rows else None
        groups[name]=fields
    # Within-scene/domain centered association, descriptive only.
    correlation={}
    for channel in CHANNELS:
        for metric in ("shared_improvement_mm2","contrast_improvement_mm2",
                       "action_relative_mae_improvement_mm"):
            k="all_"+channel+"_"+metric
            pool={}
            for row in result:
                if row[k] is not None:
                    pool.setdefault((row["scene"],row["domain"]),[]).append(row)
            x=[];y=[]
            for batch in pool.values():
                v=np.asarray([z[k] for z in batch]);q=np.asarray([z["delta_gain"] for z in batch])
                x.extend(v-v.mean());y.extend(q-q.mean())
            x=np.asarray(x);y=np.asarray(y)
            correlation[k]=float(np.corrcoef(x,y)[0,1]) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else None
    # Clustered geometry-effect intervals, not IID query/family uncertainty.
    geometry_ci={}
    for region in ("all","opening_core_proxy"):
        for channel in CHANNELS:
            for metric in ("total_improvement_mm2","shared_improvement_mm2",
                           "contrast_improvement_mm2","action_relative_mae_improvement_mm"):
                key=region+"_"+channel+"_"+metric
                available=[row for row in result if row[key] is not None]
                geometry_ci[key]=(scene_bootstrap(available,metric=key,
                    n_boot=bootstrap,seed=seed) if available else None)
    dest=new_directory(output)
    summary=dict(schema=SCHEMA,partition="development_validation",
        source_p1_fingerprint=pi["fingerprint"],source_p4_sha256=sha256(p4_summary),
        source_p5_sha256=sha256(p5_summary),n_families=len(result),n_frames=len(frames),
        strict_all_seven_action_mask=True,unknown_as_zero=False,
        same_stencil_not_same_world_point=True,subgroups=groups,
        within_scene_descriptive_correlations=correlation,
        clustered_geometry_improvement_ci=geometry_ci,
        frozen_gain_delta_ci=scene_bootstrap(result,metric="delta_gain",n_boot=bootstrap,seed=seed),
        official_ap_not_run=True,
        limitations="One-seed reused dev validation; Clean privileged; unknown sites excluded. These are not physical contact or official AP.")
    summary["artifact"]=write_records(dest/"transmission_families.jsonl.gz",result)
    atomic_json(dest/"summary.json",signed(summary))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p1-index",required=True);p.add_argument("--p4-summary",required=True)
    p.add_argument("--p5-summary",required=True);p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=5000);p.add_argument("--seed",type=int,default=0)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    r=analyze(a.p1_index,a.p4_summary,a.p5_summary,a.output,a.bootstrap,a.seed,a.smoke)
    print("P6-1 diagnostics: ",r["n_frames"],"frames",r["n_families"],"families")


if __name__=="__main__":
    main()

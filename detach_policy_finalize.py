"""CPU-only audit and tables/figures from completed detach-policy artifacts."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GROUPS = dict(A='A_all_open_r1', B='B_detach_E_r1', C='C_detach_all_r1')
METRICS = ['fg_mae_mm', 'fg_bias_mm', 'A_B_mm', 'R_B',
           'interior_adjacent_error_mm', 'edge_near_adjacent_error_mm',
           'common_uv_depth_mae_mm']


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def table(path, data):
    assert data
    keys = list(dict.fromkeys(k for r in data for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(data)


def run(root):
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    audit, aps, formations, grads, coverage, means = {}, [], [], [], [], []
    contracts, frames = [], []
    for arm, group in GROUPS.items():
        dest = root/'dev_ap'/group
        status, contract = read(dest/'status.json'), read(dest/'contract.json')
        assert status['status'] == 'complete' and not status['smoke']
        assert (root/'dev_ap'/f'{group}.exit').read_text().strip() == '0'
        assert contract['global_step'] == 17070 and contract['seed'] == 0
        assert contract['scenes'] == [100, 110, 119, 129]
        assert contract['frames'] == list(range(256))
        normalized = json.loads(json.dumps(contract))
        for k in ['arm', 'load']:
            normalized.pop(k)
        for k in ['detach_depth', 'detach_depth_gse', 'detach_depth_seed_xyz', 'detach_depth_support']:
            normalized['config'].pop(k)
        contracts.append(normalized)
        rr = list(rows(dest/'inference_frames.jsonl'))
        assert [(r['scene'], r['frame']) for r in rr] == [(s, f) for s in contract['scenes'] for f in range(256)]
        frames.append([(r['scene'], r['frame'], r['input_sha256']) for r in rr])
        for r in rr:
            path = dest/f"scene_{r['scene']:04d}"/'realsense'/f"{r['frame']:04d}.npy"
            a = np.load(path)
            assert a.shape == (r['retained'], 17) and np.isfinite(a).all()
            assert sha(path) == r['prediction_sha256']
        arr = np.load(dest/'ap_development.npy')
        assert arr.shape == (4, 256, 50, 6) and np.isfinite(arr).all()
        assert ((arr >= 0) & (arr <= 1)).all()
        assert abs(arr.mean()*100-status['AP_percent']) < 1e-10
        for i, s in enumerate(contract['scenes']):
            assert np.array_equal(arr[i], np.load(dest/f'ap_scene_{s:04d}.npy'))
            aps.append(dict(group=arm, seed=0, global_step=17070, split='test_seen_dev_4scenes',
                            scene=s, frames=256, units='percent', AP=arr[i].mean()*100,
                            AP04=arr[i, :, :, 1].mean()*100, AP08=arr[i, :, :, 3].mean()*100,
                            missing_reason=''))
        aps.append(dict(group=arm, seed=0, global_step=17070, split='test_seen_dev_4scenes',
                        scene='all', frames=1024, units='percent', AP=arr.mean()*100,
                        AP04=arr[:, :, :, 1].mean()*100, AP08=arr[:, :, :, 3].mean()*100,
                        missing_reason=''))
        audit[arm] = dict(status='pass', frames=1024, prediction_hashes_verified=1024,
                          AP_percent=float(arr.mean()*100), nonempty=sum(r['retained'] > 0 for r in rr),
                          mean_retained=float(np.mean([r['retained'] for r in rr])),
                          ap_array_sha256=sha(dest/'ap_development.npy'))
        fr = list(rows(root/group/'formation_curves.jsonl'))
        assert len({(r['global_step'], r['stem']) for r in fr}) == len(fr)
        for r in fr:
            formations.append(dict(group=arm, seed=0, global_step=r['global_step'], split=r['split'],
                scene=r['scene'], frame=r['frame'], continuity=r['continuity'], units='mm; R_B dimensionless',
                missing_reason='' if r['fg_status']=='valid' else r['fg_status'],
                **{k: r.get(k, 'NA') for k in METRICS}))
        for step in sorted({r['global_step'] for r in fr}):
            for subset in ['train4', 'seen4', 'seen32']:
                selected = [r for r in fr if r['global_step']==step and
                            (r['split']=='train' if subset=='train4' else r['split']=='test_seen') and
                            (subset=='seen32' or r['continuity'])]
                expected = 32 if subset=='seen32' else 4
                if len(selected) != expected:
                    continue
                assert all(r['fg_status']=='valid' and r['fg_finite_count']==r['fg_count'] for r in selected)
                means.append(dict(group=arm, seed=0, global_step=step, split=subset, frames=expected,
                                  units='mm; R_B dimensionless', missing_reason='',
                                  **{k: float(np.mean([r[k] for r in selected])) for k in METRICS}))
        stats = defaultdict(lambda: defaultdict(float))
        for rank in range(3):
            count = 0
            for r in rows(root/group/f'train_rank{rank}.jsonl'):
                count += 1
                assert r['global_step']==count and r['rank']==rank
                assert np.isfinite(r['global_grad_norm']) and np.isfinite(r['clip'])
                e, c = stats[r['epoch']], r['coverage']
                e['rank_batches'] += 1
                for k in ['depth_denominator', 'view_denominator', 'depth_GT_valid', 'cdf_positive']:
                    e[k] += c[k]
                for k in ['batch_grasp_cdf_valid_mask', 'batch_grasp_width_valid_mask_angle_depth', 'batch_valid_mask']:
                    for field in ['count', 'total', 'empty']:
                        e[k+'_'+field] += c[k][field]
                if rank==0:
                    e['updates'] += 1
                    e['global_grad_norm_sum'] += r['global_grad_norm']
                    e['clip_sum'] += r['clip']
                    e['clipped_updates'] += r['clip']<1
                    for scope, squared in r['postclip_gradient_squared_norm'].items():
                        assert squared>=0 and np.isfinite(squared)
                        grads.append(dict(group=arm, seed=0, global_step=count, split='train', scene='all_rank0_batch',
                            epoch=r['epoch'], scope=scope, units='parameter_gradient_L2',
                            gradient_norm=np.sqrt(squared), global_preclip_norm=r['global_grad_norm'],
                            clip=r['clip'], missing_reason='',
                            interpretation='DDP aggregate postclip; not loss-specific attribution'))
            assert count==17070
        for epoch, s in stats.items():
            coverage.append(dict(group=arm, seed=0, global_step=(epoch+1)*2845, split='train', scene='all_train',
                epoch=epoch, units='counts; ratios dimensionless', missing_reason='', **s,
                depth_valid_fraction=s['depth_GT_valid']/s['depth_denominator'],
                mean_global_preclip_norm=s['global_grad_norm_sum']/s['updates'],
                mean_clip=s['clip_sum']/s['updates']))
    assert contracts[0]==contracts[1]==contracts[2], 'Inference contracts differ beyond arm/load/detach'
    assert frames[0]==frames[1]==frames[2], 'Unpaired inference input'
    for name, data in [('ap_summary', aps), ('formation_curves', formations), ('formation_means', means),
                       ('gradient_summary', grads), ('coverage_summary', coverage)]:
        table(root/(name+'.csv'), data)
    table(root/'paired_restart.csv', [dict(group='A/B/C', seed=0, global_step='NA', split='NA', scene='NA',
        units='NA', status='skipped', missing_reason='Target epoch16 lacks matching AdamW; no P3 updates executed')])
    endpoint = {a: next(r for r in means if r['group']==a and r['global_step']==17070 and r['split']=='seen32') for a in GROUPS}
    dump(root/'final_geometry.json', endpoint)
    dump(root/'dev_ap_acceptance.json', dict(status='pass', groups=audit, all_1024_inputs_paired=True,
        common_contract_except_detach=True, scope='four prespecified test_seen development scenes; not full-split AP'))
    dump(root/'p5_decision.json', dict(status='not_triggered', recommended='C_detach_all',
        candidate=None, reason='Neither A nor B has an aggregate development AP advantage over C; C also has lower FG MAE.',
        C_minus_B_AP_percentage_points=audit['C']['AP_percent']-audit['B']['AP_percent'],
        P4_updates_per_arm=17070, P5_additional_updates=0, original_total_budget=59745,
        limitation='Single seed, six epochs, development subset; long-term/full-split superiority unverified'))
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    colors = dict(A='#ca403a', B='#377eb8', C='#238b45')
    def save(fig, name):
        fig.savefig(figures/(name+'.png'), dpi=170)
        fig.savefig(figures/(name+'.pdf'))
        plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, metric, title in zip(axs, ['fg_mae_mm','A_B_mm','interior_adjacent_error_mm'],
                               ['Foreground MAE','Target-band amplitude','Same-instance interior error']):
        for arm in GROUPS:
            rr = [r for r in means if r['group']==arm and r['split']=='seen32']
            ax.plot([r['global_step'] for r in rr], [r[metric] for r in rr], '-o', label=arm, color=colors[arm])
        ax.set(xlabel='Optimizer updates', ylabel='mm', title=title); ax.grid(alpha=.2)
    axs[0].legend(); fig.suptitle('Fixed 32-frame development set | seed 0 | common epoch endpoints')
    save(fig, 'formation_depth_geometry')
    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    for arm in GROUPS:
        rr=[r for r in means if r['group']==arm and r['split']=='seen4']
        ax.plot([r['global_step'] for r in rr], [r['A_B_mm'] for r in rr], label=arm, color=colors[arm])
    ax.axvspan(3750, 4000, color='grey', alpha=.18, label='A first crossing interval')
    ax.axhline(2, color='grey', linestyle=':'); ax.legend()
    ax.set(xlabel='Optimizer updates', ylabel='A_B (mm)', title='Original fixed 4 seen frames | periodic formation')
    save(fig, 'formation_event')
    fig, ax = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
    for i, arm in enumerate(GROUPS):
        rr=[r for r in aps if r['group']==arm]
        ax.bar(np.arange(5)+(i-1)*.25, [r['AP'] for r in rr], width=.25, label=arm, color=colors[arm])
    ax.set_xticks(range(5), ['100','110','119','129','Mean'])
    ax.set(xlabel='Development scene', ylabel='AP (%)', ylim=(0,100), title='Common 17,070-update endpoint | seed 0')
    ax.legend(); save(fig, 'development_ap')
    probes = [r for r in read(root/'probe_manifest.json')['frames'] if r['split']=='test_seen']
    fig, axs = plt.subplots(4, 8, figsize=(19, 10), constrained_layout=True)
    for i, probe in enumerate(probes):
        data=np.load(root/'inputs'/(probe['stem']+'.npz'))
        rgb=np.clip(data['rgb'].transpose(1,2,0)*[.229,.224,.225]+[.485,.456,.406],0,1)
        gt=data['gt']; valid=np.isfinite(gt)&(gt>=.2)&(gt<=1.)
        axs[i,0].imshow(rgb)
        im=axs[i,1].imshow(np.where(valid,gt,np.nan),vmin=.2,vmax=1,cmap='viridis')
        for j, (arm,group) in enumerate(GROUPS.items()):
            pred=np.load(root/group/'predictions'/f"step017070_{probe['stem']}.npz")['pred'].squeeze()
            axs[i,2+j].imshow(pred,vmin=.2,vmax=1,cmap='viridis')
            err=axs[i,5+j].imshow(np.where(valid,1000*(pred-gt),np.nan),vmin=-50,vmax=50,cmap='RdBu_r')
        axs[i,0].set_ylabel(f"{probe['scene']}\nframe {probe['frame']}")
        for ax in axs[i]:
            ax.set_xticks([]); ax.set_yticks([])
    for ax,t in zip(axs[0],['RGB','GT','A depth','B depth','C depth','A error','B error','C error']): ax.set_title(t)
    fig.colorbar(im,ax=axs[:,1:5].ravel().tolist(),location='bottom',shrink=.65,label='Depth (m), common [0.2, 1.0] display range')
    fig.colorbar(err,ax=axs[:,5:].ravel().tolist(),location='bottom',shrink=.75,extend='both',label='Pred - GT (mm), display clipped at +/-50')
    fig.suptitle('All four predeclared continuity seen frames | 17,070 updates | invalid GT blank in error maps')
    save(fig, 'shared_depth_error')
    print(json.dumps(dict(status='complete', AP=audit, coverage_rows=len(coverage), gradient_rows=len(grads))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)

#!/usr/bin/env python3
"""P0-C image-plane registration preflight for paired Original/GN-Trans RGB.

This renders GraspNet object-mask outlines on BOTH RGB domains. It can detect
missing files and mismatched image size but does not infer that two appearances
are geometrically aligned merely because their resolutions match. Inspect the
outlines for each scene before making a material-only performance claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def boundary(mask):
    # 4-neighbor object-mask boundary, avoiding a scipy/opencv dependency.
    p=np.pad(mask.astype(bool),1)
    core=p[1:-1,1:-1]
    return core & (~p[:-2,1:-1] | ~p[2:,1:-1] |
                   ~p[1:-1,:-2] | ~p[1:-1,2:])


def inspect(dataset_root,gntrans_root,out,scenes=(100,130,160),frames=(0,80,160,240),max_width=720):
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output nonempty: {out}')
    if max_width<128 or not scenes or not frames:
        raise ValueError('Choose scenes/frames and max_width>=128')
    results=[]
    for scene in scenes:
        if scene not in range(100,190):raise ValueError(f'Invalid test scene {scene}')
        for frame in frames:
            if frame not in range(0,256,10):raise ValueError(f'Invalid tested frame {frame}')
            original=dataset_root/'scenes'/f'scene_{scene:04d}'/'realsense'/'rgb'/f'{frame:04d}.png'
            gn=gntrans_root/'scenes'/f'{scene:05d}'/f'{frame:04d}_color.png'
            label=dataset_root/'scenes'/f'scene_{scene:04d}'/'realsense'/'label'/f'{frame:04d}.png'
            for p in (original,gn,label):
                if not p.is_file(): raise FileNotFoundError(f'Missing registration input {p}')
            with Image.open(original) as ia, Image.open(gn) as ib, Image.open(label) as il:
                a=np.asarray(ia.convert('RGB'),dtype=np.uint8)
                b=np.asarray(ib.convert('RGB'),dtype=np.uint8)
                seg=np.asarray(il)
            if a.shape!=b.shape or a.shape[:2]!=seg.shape:
                raise ValueError(f'RGB/label sizes differ for scene {scene}, frame {frame}: {a.shape}, {b.shape}, {seg.shape}')
            mask=seg>0
            edge=boundary(mask)
            ao,bo=a.copy(),b.copy()
            ao[edge]=[255,220,0]
            bo[edge]=[255,220,0]
            overlay=(a.astype(np.uint16)+b.astype(np.uint16))//2
            overlay=overlay.astype(np.uint8)
            overlay[edge]=[255,220,0]
            image=Image.fromarray(np.concatenate((ao,bo,overlay),axis=1),'RGB')
            if image.width>3*max_width:
                h=max(1,round(image.height*(3*max_width/image.width)))
                image=image.resize((3*max_width,h),Image.Resampling.BILINEAR)
            output=out/f'scene_{scene:04d}_frame_{frame:04d}.png'
            out.mkdir(parents=True,exist_ok=True)
            image.save(output)
            results.append({'scene':scene,'frame':frame,'original':str(original),'gntrans':str(gn),
                            'label':str(label),'width':a.shape[1],'height':a.shape[0],
                            'original_sha256':_hash(original),'gntrans_sha256':_hash(gn),
                            'montage':str(output),'boundary_pixels':int(edge.sum())})
    (out/'REGISTRATION_AUDIT.json').write_text(json.dumps({'pairs':results,'number':len(results),
        'warning':'Visual/semantic alignment MUST be reviewed manually; same pixel resolution is not proof of same geometric support.'},indent=2)+'\n')
    (out/'README.md').write_text('# P0-C input RGB spatial registration check\n\n'
        'Each montage: Original with yellow object-mask boundary | GN-Trans with SAME boundary | 50/50 RGB blend.\n'
        'Verify the object boundaries in both domains coincide with visible object silhouettes, camera framing and table/occluders.\n'
        'Size/paths are checked automatically; pixel-wise alignment is NOT certified by this program.\n'
        'Do NOT proceed with material-only claims if rendered objects have moved or camera parameters differ.\n')
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--gntrans-rgb-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--scenes',nargs='+',type=int,default=[100,130,160,167,174])
    p.add_argument('--frames',nargs='+',type=int,default=[0,80,160,240])
    a=p.parse_args();rows=inspect(a.dataset_root,a.gntrans_rgb_root,a.output_dir,a.scenes,a.frames)
    print(json.dumps({'registered_size_pairs':len(rows),'output':str(a.output_dir),'manual_review_required':True},indent=2))

if __name__=='__main__':main()

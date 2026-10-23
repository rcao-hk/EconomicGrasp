"""Validate all six official AP arrays and dump counts before marking complete."""
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
thresholds = [.2, .4, .6, .8, 1., 1.2]
rows = []
for domain in ('original', 'gntrans'):
    for split in ('test_seen', 'test_similar', 'test_novel'):
        folder = root / 'eval_e19' / domain / split
        dumps = list(folder.glob('scene_*/realsense/*.npy'))
        assert len(dumps) == 780, (folder, len(dumps))
        result = np.load(folder / f'ap_{split}_realsense.npy')
        assert result.shape == (30, 26, 50, 6), (folder, result.shape)
        assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
        rows.append(dict(dataset=domain, split=split, frames=len(dumps),
                         AP_percent=float(result.mean()*100),
                         AP_by_mu_percent=dict(zip(map(str, thresholds),
                             (result.mean(axis=(0, 1, 2))*100).tolist()))))
report = dict(checkpoint=json.loads((root/'train_complete.json').read_text()),
              friction_coefficients=thresholds, metrics=rows)
(root/'evaluation_complete.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

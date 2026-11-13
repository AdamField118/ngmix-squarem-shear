import os
import numpy as np
import yaml

from squarem_shear.run import run_chunk
from squarem_shear.collect import summarize

HERE = os.path.dirname(__file__)


def test_smoke():
    with open(os.path.join(HERE, '..', 'configs', 'smoke.yaml')) as f:
        conf = yaml.safe_load(f)
    conf['psf']['npool'] = 2
    out = run_chunk(conf, chunk=0, nobj=2, verbose=False)
    out['config'] = np.array(yaml.safe_dump(conf))
    s = summarize(out, conf['shear']['g'], reference='em5_converged',
                  nboot=10)
    assert s['ngal'] >= 1
    for arm in conf['arms']:
        v = s['arms'][arm['name']]
        assert np.isfinite(v['m'])
        assert np.isfinite(v['R11'])

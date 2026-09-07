"""Exercise unmodified production CLIs on isolated synthetic Zarr stores."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import torch
import xarray as xr
ROOT=Path(os.environ.get('NAO_PROJECT_ROOT',Path.cwd())).resolve()
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from configs.download_data_config import channel_names
from neural_atmosphere_operator.pipeline.runtime import build_loader,dataset_signature
from configs.pipeline_config import DataPaths
EVIDENCE=Path(os.environ.get('NAO_AUDIT_EVIDENCE','/tmp/nao-audit-evidence'))
WORK=Path(tempfile.mkdtemp(prefix='nao-audit-integration-'))
results={'work_directory':str(WORK)}
names=channel_names()
units=['m s**-1','m s**-1','K','Pa','Pa','kg m**-2']+['m**2 s**-2']*5+['m s**-1']*8+['K']*4+['kg kg**-1']*2+['1']
def make_store(path,date,count=24,cadence=6,lat=None,lon=None,transpose=False):
    state=np.random.default_rng(17).normal(size=(count,26,13,24)).astype('float32')
    ds=xr.Dataset({'state':(('time','channel','latitude','longitude'),state)},coords={'time':np.datetime64(date,'h')+np.arange(count)*np.timedelta64(cadence,'h'),'channel':list(names),'channel_units':('channel',units),'latitude':np.linspace(90,-90,13) if lat is None else lat,'longitude':np.arange(24)*15. if lon is None else lon})
    if transpose: ds=ds.transpose('time','latitude','channel','longitude')
    ds.chunk({'time':1}).to_zarr(str(path),mode='w',consolidated=True)
make_store(WORK/'train.zarr','1995-01-01')
make_store(WORK/'valid.zarr','2019-01-01')
make_store(WORK/'cadence12.zarr','2019-01-01',cadence=12)
make_store(WORK/'shifted.zarr','2019-01-01',lon=np.arange(24)*15.+15.)
make_store(WORK/'transposed.zarr','2019-01-01',transpose=True)
stats=WORK/'stats'
def run(name,args,code=None):
    env={**os.environ,'OMP_NUM_THREADS':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'}
    command=[sys.executable]+(['-c',code]+args if code else args)
    p=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
    (EVIDENCE/(name+'.txt')).write_text(p.stdout+p.stderr)
    results[name]={'exit_code':p.returncode}
    if p.returncode: results[name]['error_tail']=(p.stdout+p.stderr)[-700:]
    return p.returncode
assert run('integration_stats',['scripts/compute_stats.py','--train-data',str(WORK/'train.zarr'),'--output-dir',str(stats)])==0
base=['--train-data',str(WORK/'train.zarr'),'--valid-data',str(WORK/'valid.zarr'),'--means-path',str(stats/'means.npy'),'--stds-path',str(stats/'stds.npy'),'--time-diff-stds-path',str(stats/'time_diff_stds_dt1.npy'),'--epochs','2','--scheduler-epochs','5','--warmup-epochs','2','--batch-size','4','--gradient-accumulation','2','--rollout-steps','1','--validation-rollout-steps','2','--num-workers','0','--embed-dim','8','--num-layers','2','--device','cpu','--max-samples','17']
snapshot=WORK/'epoch0.pt'
wrapper="import sys,runpy,shutil; sys.path[:0]=['.','src']; import neural_atmosphere_operator.pipeline.checkpoint as cp; original=cp.save_checkpoint\ndef save(path,**kw):\n original(path,**kw)\n if kw['epoch']==0 and path.name=='last.pt': shutil.copyfile(path,"+repr(str(snapshot))+")\ncp.save_checkpoint=save\nrunpy.run_path('scripts/train.py',run_name='__main__')"
assert run('integration_full',base+['--run-dir',str(WORK/'full')],wrapper)==0
assert run('integration_resume',['scripts/train.py']+base+['--run-dir',str(WORK/'resume'),'--resume',str(snapshot)])==0
def compare(left,right):
    a=torch.load(left,weights_only=False,map_location='cpu');b=torch.load(right,weights_only=False,map_location='cpu')
    return max(float((a['model_state'][k]-b['model_state'][k]).abs().max()) for k in a['model_state'])
full=WORK/'full/checkpoints/last.pt'
results['same_config_resume_weight_max_error']=compare(full,WORK/'resume/checkpoints/last.pt')
assert run('integration_changed_warmup',['scripts/train.py']+base+['--run-dir',str(WORK/'changed_warmup'),'--resume',str(snapshot),'--warmup-start-factor','0.9'])==0
results['changed_warmup_resume_weight_max_error']=compare(full,WORK/'changed_warmup/checkpoints/last.pt')
assert run('integration_changed_sample_count',['scripts/train.py']+base+['--run-dir',str(WORK/'changed_samples'),'--resume',str(snapshot),'--max-samples','18'])==0
results['changed_sample_resume_weight_max_error']=compare(full,WORK/'changed_samples/checkpoints/last.pt')
ev=['scripts/evaluate.py','--checkpoint',str(full),'--means-path',str(stats/'means.npy'),'--stds-path',str(stats/'stds.npy'),'--climatology-path',str(stats/'time_means.npy'),'--num-workers','0','--device','cpu','--rollout-steps','2','--max-samples','1']
for name,path in [('evaluation_valid','valid.zarr'),('evaluation_wrong_cadence','cadence12.zarr'),('evaluation_shifted_longitude','shifted.zarr'),('evaluation_training_as_test','train.zarr')]:
    assert run(name,ev+['--data-path',str(WORK/path),'--output-dir',str(WORK/name),'--split','test'])==0
    result=json.loads((WORK/name/'metrics.json').read_text());results[name]['lead_hours']=result['metrics'][0]['lead_hours'];results[name]['data_first_time']=result['data_first_time']
assert run('integration_rollout',['scripts/rollout.py','--checkpoint',str(full),'--data-path',str(WORK/'valid.zarr'),'--means-path',str(stats/'means.npy'),'--stds-path',str(stats/'stds.npy'),'--rollout-steps','2','--device','cpu','--output',str(WORK/'forecast.zarr')])==0

from neural_atmosphere_operator.data.loader import AtmosphereDatasetConfig,AtmosphereZarrDataset
d=AtmosphereZarrDataset(AtmosphereDatasetConfig(data_path=WORK/'transposed.zarr',normalize=False))
results['transposed_state_accepted_input_shape']=list(d[0]['input'].shape);d.close()
# Regional grid accepted in eval path through spatial_crop despite matching shape.
make_store(WORK/'regional.zarr','2019-01-01',lat=np.linspace(20,0,13),lon=np.arange(24))
assert run('evaluation_regional',ev+['--data-path',str(WORK/'regional.zarr'),'--output-dir',str(WORK/'evaluation_regional')])==0
print(json.dumps(results,indent=2))
(EVIDENCE/'integration.json').write_text(json.dumps(results,indent=2)+'\n')

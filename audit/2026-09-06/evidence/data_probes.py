"""Additional bounded data, spectrum, and memory probes."""
import json, os, sys, tempfile
from pathlib import Path
ROOT=Path(os.environ.get('NAO_PROJECT_ROOT',Path.cwd())).resolve()
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import xarray as xr
import torch
from dask.callbacks import Callback
from scripts.compute_stats import spherical_channel_statistics
from neural_atmosphere_operator.data.normalization import latitude_cell_weights
from data.download_data import _conservative_half_degree,_latitude_overlap_weights,build_subset
from configs.download_data_config import WeatherBenchDownloadConfig
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator
from configs.model_config import AtmosphereModelConfig

torch.set_num_threads(1)
out={}
folder=Path(tempfile.mkdtemp(prefix='nao-audit-data-'))
for n in [32,128]:
    state=np.random.default_rng(4).normal(size=(n,2,13,24)).astype('float32')
    ds=xr.Dataset({'state':(('time','channel','latitude','longitude'),state)},coords={'latitude':np.linspace(90,-90,13)})
    ds.chunk({'time':1}).to_zarr(str(folder/f'{n}.zarr'),consolidated=True)
    del state,ds
    ds=xr.open_zarr(str(folder/f'{n}.zarr'),consolidated=True)
    w=xr.DataArray(np.asarray(latitude_cell_weights(ds.latitude.values),dtype=np.float64),dims=('latitude',),coords={'latitude':ds.latitude})
    peak=[0]
    def post(key,result,dsk,state,worker):
        size=0;seen=set()
        for v in state['cache'].values():
            if isinstance(v,np.ndarray):
                while isinstance(v.base,np.ndarray): v=v.base
                if id(v) not in seen: size+=v.nbytes;seen.add(id(v))
        peak[0]=max(peak[0],size)
    with Callback(posttask=post): spherical_channel_statistics(ds.state,w,1).compute(scheduler='single-threaded')
    out[f'zarr_dask_cache_{n}']={'input_bytes':n*2*13*24*4,'peak_bytes':peak[0],'ratio':peak[0]/(n*2*13*24*4)}
    ds.close()

# The 3-point conservative longitude filter is not a strict anti-alias filter.
rows=_latitude_overlap_weights(np.linspace(90,-90,721),np.linspace(90,-90,361))
wave=np.cos(2*np.pi*500*np.arange(1440)/1440).astype('float32')
y=_conservative_half_degree(np.broadcast_to(wave,(721,1440)),rows)[180]
spectrum=np.abs(np.fft.rfft(y))*2/len(y)
out['downsample_alias']={'source_mode':500,'source_grid_width':1440,'target_grid_width':720,'output_dominant_mode':int(spectrum.argmax()),'output_amplitude':float(spectrum.max())}

# Same-sized, different latitude nodes still require residual resampling.
m=AtmosphereNeuralOperator(AtmosphereModelConfig(img_size=(13,24),scale_factor=1,embed_dim=4,num_layers=2,in_channels=1,out_channels=1))
conv=m.backbone.blocks[0].global_conv
x=torch.cos(torch.linspace(0,torch.pi,13))[:,None].expand(13,24)[None,None].contiguous()
with torch.no_grad():
    conv.weight.zero_()
    _,res=conv(x.expand(1,4,13,24))
    expected=conv.inverse_transform(conv.forward_transform(x.expand(1,4,13,24)))
out['scale1_residual']={'input_grid':conv.forward_transform.grid,'output_grid':conv.inverse_transform.grid,'scale_residual':conv.scale_residual,'residual_error_max':float((res-expected).abs().max())}

# Each seven-day batch resets cadence selection; 18-hour stride is accepted.
base=np.datetime64('2018-01-01T00','h')
native=base+np.arange(14*4)*np.timedelta64(6,'h')
one=native[:28][::3];two=native[28:][::3]
combined=np.concatenate([one,two])
WeatherBenchDownloadConfig(time_stride_hours=18)
out['batched_18h_cadence']={'delta_hours':np.unique(np.diff(combined).astype('timedelta64[h]').astype(int)).tolist(),'boundary_gap_hours':int((two[0]-one[-1])/np.timedelta64(1,'h'))}
out['full_grid_bytes']={'state':26*361*720*4,'training_corpus':35064*26*361*720*4,'valid_60_targets':60*26*361*720*4,'test_120_targets':120*26*361*720*4}
print(json.dumps(out,indent=2))
Path('/tmp/nao-audit-evidence/data-probes.json').write_text(json.dumps(out,indent=2)+'\n')

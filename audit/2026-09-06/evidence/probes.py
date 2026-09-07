"""Independent scientific audit probes; no production files are modified."""
import ast
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(os.environ.get('NAO_PROJECT_ROOT', Path.cwd())).resolve()
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import torch
import xarray as xr
import dask.array as da
from dask import delayed
from dask.callbacks import Callback
from configs.model_config import AtmosphereModelConfig
from configs.pipeline_config import DataPaths
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator
from neural_atmosphere_operator.models.loss import ChannelRelativeAtmosphereLoss, latitude_weighted_mse, LossScaler
from neural_atmosphere_operator.pipeline.schedule import create_warmup_cosine_scheduler
from neural_atmosphere_operator.pipeline.runtime import build_loader, dataset_signature
from neural_atmosphere_operator.pipeline.forecast import rollout_loss
from neural_atmosphere_operator.data.normalization import latitude_cell_weights, AtmosphereNormalizer
from scripts.compute_stats import spherical_channel_statistics
from torch_harmonics import RealSHT, InverseRealSHT
from torch_harmonics.examples.models._layers import SpectralConvS2

torch.set_num_threads(1)
torch.manual_seed(42)
results = {}
def record(name, **data):
    results[name] = data
    print(name, json.dumps(data, default=str), flush=True)

def cfg(**kwargs):
    base = dict(img_size=(13,24), in_channels=4, out_channels=4, scale_factor=3, embed_dim=8, num_layers=2)
    base.update(kwargs)
    return AtmosphereModelConfig(**base)

# Exact prediction must have a finite zero subgradient for a robust relative norm.
p = torch.ones(1,2,5,8, requires_grad=True)
l,_ = ChannelRelativeAtmosphereLoss()(p, p.detach().clone())
l.backward()
record('relative_loss_at_perfect_prediction', loss=l.item(), finite_gradient=bool(p.grad.isfinite().all()), nan_count=int(p.grad.isnan().sum()))

# Match the default train.py step ordering with disabled BF16 GradScaler.
p = torch.nn.Parameter(torch.tensor([1.]))
opt = torch.optim.AdamW([p],lr=1e-3)
scaler = torch.amp.GradScaler('cuda', enabled=False)
scaler.scale((p*float('nan')).sum()).backward()
scaler.unscale_(opt)
norm = torch.nn.utils.clip_grad_norm_([p],1.)
scaler.step(opt); scaler.update()
record('nonfinite_default_optimizer_step', preclip_norm=float(norm), parameter_finite=bool(p.isfinite().all()))

# Constructor accepts real-kernel switch but upstream ignores it.
torch.manual_seed(10); a=AtmosphereNeuralOperator(cfg(use_complex_kernels=True))
torch.manual_seed(10); b=AtmosphereNeuralOperator(cfg(use_complex_kernels=False))
x=torch.randn(1,4,13,24)
record('complex_flag', parameters_identical=all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters())), outputs_identical=torch.equal(a(x),b(x)), spectral_dtype=str(b.backbone.blocks[0].global_conv.weight.dtype))

# Analytic rotation counterexample: z=cos(theta) rotates to x=sin(theta)cos(phi).
sht=RealSHT(33,64,lmax=8,mmax=8).float()
isht=InverseRealSHT(33,64,lmax=8,mmax=8).float()
conv=SpectralConvS2(sht,isht,1,1,operator_type='driscoll-healy')
th=torch.linspace(0,math.pi,33)[:,None]
ph=torch.arange(64)[None,:]*(2*math.pi/64)
z=th.cos().expand(33,64)[None,None].contiguous()
xx=(th.sin()*ph.cos())[None,None].contiguous()
with torch.no_grad():
    conv.weight.fill_(1j)
    kz=conv(z)[0]; kx=conv(xx)[0]
    complex_result=dict(K_z_max=float(kz.abs().max()), K_rotated_z_max=float(kx.abs().max()))
    conv.weight.fill_(1.)
    real_error=max(float((conv(z)[0]-z).abs().max()),float((conv(xx)[0]-xx).abs().max()))
    roundtrip=max(float((isht(sht(z))-z).abs().max()),float((isht(sht(xx))-xx).abs().max()))
record('degree1_rotation_counterexample', **complex_result, real_kernel_identity_error=real_error, sht_roundtrip_error=roundtrip)

# LayerNorm one-block shape and grid-conversion residual behavior.
try:
    AtmosphereNeuralOperator(cfg(num_layers=1,normalization_layer='layer_norm'))(x)
    record('one_layer_layer_norm', rejected=False)
except Exception as e: record('one_layer_layer_norm', rejected=True,error=str(e))
record('scale1_grid_conversion', scale_residual=AtmosphereNeuralOperator(cfg(scale_factor=1)).backbone.blocks[0].global_conv.scale_residual)

# Kernel l=0 imaginary values never contribute to any real scalar field.
a=AtmosphereNeuralOperator(cfg())
y=a(x); y.square().mean().backward()
record('gradient_coverage', tensors_without_gradient=[n for n,p in a.named_parameters() if p.grad is None], all_gradients_finite=all(p.grad is None or bool(p.grad.isfinite().all()) for p in a.parameters()), degree_zero_imaginary_gradient_max=max(float(b.global_conv.weight.grad[...,0].imag.abs().max()) for b in a.backbone.blocks))

# Finite-difference derivative of model along a random input direction.
a.eval(); x=x.detach().requires_grad_(); direction=torch.randn_like(x); direction/=direction.norm()
y=a(x).square().mean(); grad=torch.autograd.grad(y,x)[0]; ad=float((grad*direction).sum())
eps=0.01
with torch.no_grad(): fd=float((a(x+eps*direction).square().mean()-a(x-eps*direction).square().mean())/(2*eps))
record('model_directional_derivative',autograd=ad,finite_difference=fd,absolute_error=abs(ad-fd))

# Weighted MSE double-precision autograd check; custom LossScaler is deliberately non-derivative.
v=torch.randn(1,2,3,4,dtype=torch.double,requires_grad=True); t=torch.randn_like(v)
record('weighted_mse_gradcheck', passed=torch.autograd.gradcheck(lambda v: latitude_weighted_mse(v,t), (v,)))
try:
    torch.autograd.gradcheck(lambda v: (LossScaler()(v)*v.detach().new_tensor([1.,10.]).view(1,2,1,1)).sum(),(v,))
    record('loss_scaler_gradcheck',passed=True)
except Exception: record('loss_scaler_gradcheck',passed=False,expected='Custom backward is not derivative of identity forward')

# Gradient checkpointing should preserve values and gradient for two-step BPTT.
aa=AtmosphereNeuralOperator(cfg()); bb=copy.deepcopy(aa)
batch={'input':torch.randn(1,4,13,24),'target':torch.randn(1,2,4,13,24)}
la,_,_=rollout_loss(aa,batch,2,gradient_checkpointing=False); la.backward()
lb,_,_=rollout_loss(bb,batch,2,gradient_checkpointing=True); lb.backward()
record('checkpoint_gradient_equivalence',loss_error=float((la-lb).abs()),gradient_max_error=max(float((p.grad-q.grad).abs().max()) for p,q in zip(aa.parameters(),bb.parameters())))

# train/eval, zero, constant and short synthetic overfit.
a=AtmosphereNeuralOperator(cfg()); a.train(); yt=a(batch['input']); a.eval(); ye=a(batch['input'])
with torch.no_grad():
    zz=a(torch.zeros_like(batch['input'])); cc=a(torch.ones_like(batch['input']))
record('model_basic_inputs',train_eval_max_error=float((yt-ye).abs().max()),zero_output_max=float(zz.abs().max()),constant_output_finite=bool(cc.isfinite().all()),constant_spatial_std=float(cc.std(dim=(-2,-1)).max()))
opt=torch.optim.AdamW(a.parameters(),lr=3e-3)
target=batch['input']+0.1*torch.sin(batch['input'])
initial=float(latitude_weighted_mse(a(batch['input']),target))
for _ in range(150):
    opt.zero_grad(); l=latitude_weighted_mse(a(batch['input']),target); l.backward(); opt.step()
final=float(latitude_weighted_mse(a(batch['input']),target))
record('synthetic_one_sample_overfit',updates=150,initial_loss=initial,final_loss=final,ratio=final/initial)
with torch.no_grad():
    state=batch['input']; norms=[]
    for k in range(120):
        state=a(state)
        if k in [0,3,19,59,119]: norms.append([k+1,float(state.square().mean().sqrt()),bool(state.isfinite().all())])
record('synthetic_rollout_120',lead_rms_finite=norms,qualification='Toy fitted map, not atmospheric skill or stability evidence')

# LambdaLR state omits lambda closures: altered warmup start changes next update.
def scheduler(factor):
    p=torch.nn.Parameter(torch.tensor([1.]))
    o=torch.optim.AdamW([p],lr=1e-3)
    s=create_warmup_cosine_scheduler(o,total_updates=20,warmup_updates=10,warmup_start_factor=factor,min_learning_rate=1e-6,base_learning_rate=1e-3)
    return o,s
o,s=scheduler(.1)
for _ in range(2): o.step(); s.step()
oo,ss=scheduler(.9); oo.load_state_dict(o.state_dict()); ss.load_state_dict(s.state_dict())
o.step();s.step();oo.step();ss.step()
tree=ast.parse((ROOT/'scripts/train.py').read_text())
keys=[]
for n in ast.walk(tree):
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='resume_signature' for t in n.targets): keys=[k.value for k in n.value.keys]
record('resume_warmup_change',checked_in_signature='warmup_start_factor' in keys,uninterrupted_next_lr=o.param_groups[0]['lr'],resumed_changed_next_lr=oo.param_groups[0]['lr'])

# Dask centered-variance graph cache scales with all input chunks.
@delayed
def generated_chunk(k): return np.random.default_rng(k).normal(size=(1,2,13,24)).astype(np.float32)
for n in [16,64,256]:
    ar=da.concatenate([da.from_delayed(generated_chunk(i),shape=(1,2,13,24),dtype=np.float32) for i in range(n)],axis=0)
    st=xr.DataArray(ar,dims=('time','channel','latitude','longitude'),coords={'latitude':np.linspace(90,-90,13)})
    w=xr.DataArray(np.asarray(latitude_cell_weights(st.latitude.values),dtype=np.float64),dims=('latitude',),coords={'latitude':st.latitude})
    peak=[0]
    def post(key,result,dsk,state,worker):
        seen=set(); size=0
        for val in state['cache'].values():
            if isinstance(val,np.ndarray):
                while isinstance(val.base,np.ndarray): val=val.base
                if id(val) not in seen: size+=val.nbytes;seen.add(id(val))
        peak[0]=max(peak[0],size)
    with Callback(posttask=post): spherical_channel_statistics(st,w,1).compute(scheduler='single-threaded')
    record('dask_cache_'+str(n),input_bytes=n*2*13*24*4,peak_cached_array_bytes=peak[0],ratio=peak[0]/(n*2*13*24*4))

Path(os.environ.get('NAO_AUDIT_RESULT','/tmp/nao-audit-evidence/probes.json')).write_text(json.dumps(results,indent=2,default=str)+'\n')

"""Measured callback timings with explicit units, boundaries and memory semantics."""

import platform
try:
    import resource
except ImportError:  # Windows has no stdlib resource module.
    resource = None
import time

import numpy as np
import torch


def profile_stage(callback,*,name,unit,iterations=20,warmup=3,device='cpu',mode='inference'):
    if (not name or not unit or type(iterations) is not int or iterations<1
            or type(warmup) is not int or warmup<0 or mode not in ('inference','training')):
        raise ValueError('explicit stage/unit, valid iteration counts and execution mode required')
    target=torch.device(device)
    if target.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is unavailable')
    def synchronize():
        if target.type=='cuda':torch.cuda.synchronize(target)
    context=torch.inference_mode if mode=='inference' else torch.enable_grad
    with context():
        for _ in range(warmup):callback()
        synchronize()
        baseline=None
        if target.type=='cuda':
            baseline=torch.cuda.memory_allocated(target)
            torch.cuda.reset_peak_memory_stats(target)
        timings=[]
        for _ in range(iterations):
            synchronize();start=time.perf_counter_ns()
            callback()
            synchronize();timings.append((time.perf_counter_ns()-start)/1e6)
    peak=None if target.type!='cuda' else torch.cuda.max_memory_allocated(target)
    # Linux ru_maxrss is a process-lifetime high-water mark, not incremental stage allocation.
    return {'stage':name,'unit':unit,'mode':mode,'iterations':iterations,'warmup':warmup,
            'milliseconds':{'median':float(np.median(timings)),'p90':float(np.quantile(timings,.9)),
                            'p95':float(np.quantile(timings,.95)),'samples':timings},
            'device':str(target),'torch_threads':torch.get_num_threads(),
            'python':platform.python_version(),'torch':str(torch.__version__),
            'process_lifetime_peak_rss_kib':None if resource is None else (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if platform.system()=='Darwin' else 1)),
            'cuda_baseline_allocated_bytes':baseline,'cuda_peak_allocated_bytes':peak,
            'cuda_peak_excess_bytes':None if peak is None else peak-baseline,
            'boundary_note':'only supplied callback; do not label model-only timing end-to-end'}


def parameter_storage(model):
    return {'total_parameters':sum(value.numel() for value in model.parameters()),
            'trainable_parameters':sum(value.numel() for value in model.parameters() if value.requires_grad),
            'parameter_bytes':sum(value.numel()*value.element_size() for value in model.parameters()),
            'buffer_bytes':sum(value.numel()*value.element_size() for value in model.buffers()),
            'note':'tensor storage only; not optimizer, activations or full runtime memory'}

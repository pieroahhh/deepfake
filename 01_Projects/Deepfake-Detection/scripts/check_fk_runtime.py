"""Validate FK pretrained assets on synthetic inputs; no accuracy claim."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from deepfake_detection.cli import load_json, save
from deepfake_detection.commands import load_fk_assets, fk_asset_hashes
from deepfake_detection.model import EvidenceModel

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--assets',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
torch.set_num_threads(2)
torch.manual_seed(7)
assets=load_json(args.assets)
encoders,detector,vad=load_fk_assets(assets)
features=encoders.encode_window(torch.rand(50,3,224,224),
    .1*torch.sin(2*torch.pi*440*torch.arange(32000)/16000),torch.arange(50)/25)
model=EvidenceModel()
output=model(**features,quality=torch.tensor([[.5,.5]]))
output['window_scores'].sum().backward()
assert all(p.grad is None and not p.requires_grad for p in encoders.parameters())
assert model.gru.weight_ih_l0.grad is not None
assert torch.isfinite(output['bin_scores']).all()
report={'status':'PASSED','data_kind':'synthetic_pixels_and_tone_actual_pretrained_encoders',
        'classifier':'random_untrained_FK_head','torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
        'assets':fk_asset_hashes(assets,encoders),'feature_shapes':{k:list(v.shape) for k,v in features.items()},
        'frozen_encoder_gradient_check':'PASSED','head_backward':'PASSED',
        'blank_image_detections':len(detector(np.zeros((320,320,3),dtype=np.uint8))),
        'silence_vad':vad(np.zeros(32000,dtype=np.float32))}
save(args.output,report)
print(json.dumps(report,indent=2))

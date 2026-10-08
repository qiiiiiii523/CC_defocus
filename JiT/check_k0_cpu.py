"""CPU-only K0 interface tests with a tiny substitute, not JiT training."""
import argparse
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType

import numpy as np
from PIL import Image
import torch
from dataset_cnseg import CNSegPairedDataset, gaussian_blur
from nucleus_losses import nucleus_supervision_loss


class Tiny(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(.2))
    def forward(self, z, t, labels, blur, degradation):
        return blur * .5 + self.bias


def main():
    root = Path(__file__).resolve().parent
    for name in ("main_restoration.py", "engine_restoration.py", "denoiser.py",
                 "dataset_cnseg.py", "evaluate_k0.py", "nucleus_losses.py"):
        compile((root/name).read_text(encoding="utf-8-sig"), name, "exec")
    # Do not import the actual CUDA-only JiT model on CPU.
    stub = ModuleType("model_jit"); stub.JiT_models = {"tiny": Tiny}
    sys.modules["model_jit"] = stub
    from denoiser import Denoiser
    args = SimpleNamespace(model="tiny",img_size=8,class_num=1000,attn_dropout=0.,proj_dropout=0.,
        label_drop_prob=0.,P_mean=-.8,P_std=.8,t_eps=.05,noise_scale=1.,lambda_pix=1.,
        charbonnier_eps=.001,ema_decay1=.9999,ema_decay2=.9996,sampling_method="euler",
        num_sampling_steps=1,cfg=1.,interval_min=0.,interval_max=1.,lambda_nucleus=0.)
    model = Denoiser(args)
    clear,blur = torch.zeros(1,3,8,8),torch.zeros(1,3,8,8)
    mask = torch.zeros(1,8,8,dtype=torch.int64); mask[:,2:6,2:6]=1
    torch.manual_seed(3)
    base = model(clear,blur,return_loss_components=True)
    keys = set(model.state_dict())
    model.lambda_nucleus = .5
    torch.manual_seed(3)
    result = model(clear,blur,instance_mask=mask,return_loss_components=True)
    assert torch.allclose(result['loss']-base['loss'], result['loss_nucleus_weighted'])
    assert torch.allclose(result['loss_nucleus_region'], torch.tensor(.1))
    assert set(model.state_dict()) == keys
    model.load_state_dict(model.state_dict(),strict=True)
    result['loss'].backward()
    assert torch.isfinite(model.net.bias.grad)
    empty = nucleus_supervision_loss(torch.ones(1,3,8,8,requires_grad=True),clear,torch.zeros_like(mask))
    assert empty['total'].item()==0; empty['total'].backward()
    try:
        model(clear,blur)
    except ValueError:
        pass
    else:
        raise AssertionError('missing mask not rejected')
    assert torch.allclose(gaussian_blur(torch.ones(3,8,8),dict(kernel_type='gaussian',padding='reflect',kernel_size=3,sigma=1.)),torch.ones(3,8,8))
    with tempfile.TemporaryDirectory(prefix='k0-check-') as tmp:
        folder = Path(tmp)
        image=np.zeros((12,12,3),dtype=np.uint8); image[3:9,3:9]=255
        labels=np.zeros((12,12),dtype=np.uint16); labels[3:9,3:9]=7
        Image.fromarray(image).save(folder/'clear.png'); Image.fromarray(labels).save(folder/'mask.png')
        row=dict(sample_id='sample',split='val',quality_status='pass',group_id='group',
            clear_path='clear.png',instance_mask_path='mask.png',degradation=dict(kernel_type='gaussian',padding='reflect',kernel_size=3,sigma=1.))
        (folder/'manifest.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
        dataset=CNSegPairedDataset(folder/'manifest.jsonl',split='val',image_size=8,training=False,root=folder)
        item=dataset[0]
        assert torch.equal(item['instance_mask']>0,item['clear'][0]>0)
        assert set(item['instance_mask'].unique().tolist())=={0,7}
        assert torch.equal(item['blur'],dataset[0]['blur'])
        row['split']='train'; (folder/'manifest.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
        dataset=CNSegPairedDataset(folder/'manifest.jsonl',split='train',image_size=8,training=True,identity_ratio=1,root=folder)
        for _ in range(3):
            item=dataset[0]
            assert torch.equal(item['blur'],item['clear'])
            assert torch.equal(item['instance_mask']>0,item['clear'][0]>0)
        duplicate=dict(row,split='val',sample_id='leak')
        (folder/'manifest.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(duplicate)+'\n',encoding='utf-8')
        try:
            CNSegPairedDataset(folder/'manifest.jsonl',split='train',image_size=8,training=True,root=folder)
        except ValueError as exc:
            assert 'leakage' in str(exc)
        else:
            raise AssertionError('split leakage not rejected')
    tree=ast.parse((root/'main_restoration.py').read_text(encoding='utf-8-sig'))
    parser_func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_args_parser')
    namespace=dict(argparse=argparse,Path=Path,PROJECT_ROOT=root.parent,JIT_ROOT=root)
    exec(compile(ast.Module(body=[parser_func],type_ignores=[]),'parser','exec'),namespace)
    parsed=namespace['get_args_parser']().parse_args(['--dataset_mode','cnseg','--lambda_nucleus','.5','--init_checkpoint','a0.pth'])
    assert parsed.lambda_nucleus==.5 and parsed.resume is None
    print('PASS: syntax, parser, nucleus toggle/scaling/gradient, checkpoint keys, Gaussian constant, synchronized crop/flip/identity, deterministic validation, empty mask, split leakage')


if __name__=='__main__':
    main()

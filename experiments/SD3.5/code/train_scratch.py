"""Minimal SD3.5 scratch training loop; Transformer starts from config only."""
import argparse, json, random, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model import SD35ScratchRestorer
from data import PairedManifest
from vae_utils import load_frozen_vae, encode
from flow_matching import make_training_state

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--repo-root',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--model-id',default='stabilityai/stable-diffusion-3.5-medium'); a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8')); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise SystemExit('CUDA is required for training.')
    seed=cfg.get('seed',20261010); torch.manual_seed(seed); random.seed(seed); device=torch.device('cuda'); dtype=torch.bfloat16 if cfg.get('bf16',True) else torch.float32
    model=SD35ScratchRestorer(a.model_id).to(device); vae=load_frozen_vae(a.model_id,device,dtype)
    ds=PairedManifest(Path(a.repo_root)/cfg['train_manifest'],a.repo_root,cfg.get('resolution',256),True); dl=DataLoader(ds,batch_size=cfg.get('batch_size',1),shuffle=True,num_workers=2,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.get('learning_rate',1e-5),weight_decay=1e-2); model.train(); step=0; started=time.time()
    while step<cfg.get('max_train_steps',1000):
        for batch in dl:
            with torch.no_grad(): zb=encode(vae,batch['blur'].to(device,dtype=dtype)); zc=encode(vae,batch['clear'].to(device,dtype=dtype))
            xt,cond,t,target=make_training_state(zb,zc,cfg.get('flow_condition_noise_alpha',.15)); pred=model(xt,cond,t); loss=(pred-target).float().square().mean(); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); step+=1
            if step%100==0: print(f'step={step} loss={loss.item():.6f}')
            if step%500==0: torch.save({'model':model.state_dict(),'step':step,'config':cfg},out/f'checkpoint-{step}.pt')
            if step>=cfg.get('max_train_steps',1000): break
    print(f'finished steps={step} seconds={time.time()-started:.1f}')

if __name__=='__main__': main()

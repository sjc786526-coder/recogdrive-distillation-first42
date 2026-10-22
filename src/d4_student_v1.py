"""Frozen native Base with broad LoRA and two trainable projections.

Original BF16 projection arithmetic is retained with FP32 master parameters.
All downstream sampling stays native FP32/eval, with gradients enabled. This
module never imports the inference context cache or teacher repair operators.
"""
from d4_common import *
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self,original,rank,alpha):
        super().__init__();self.base=original;self.scale=float(alpha)/rank
        self.A=nn.Parameter(torch.empty(rank,original.in_features,device=original.weight.device,dtype=torch.float32))
        self.B=nn.Parameter(torch.zeros(original.out_features,rank,device=original.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))

    def forward(self,value):
        native=self.base(value)
        delta=F.linear(F.linear(value.float(),self.A),self.B)*self.scale
        return (native.float()+delta).to(native.dtype)


class NativeProjectionMasters(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original
        names=list(original.named_parameters());self.names=[n for n,_ in names];self.dtypes=[p.dtype for _,p in names]
        self.masters=nn.ParameterList([nn.Parameter(p.detach().float().clone()) for _,p in names])

    def forward(self,*args,**kwargs):
        replacements={name:p.to(dtype) for name,p,dtype in zip(self.names,self.masters,self.dtypes)}
        return torch.func.functional_call(self.original,replacements,args,kwargs,strict=False)


class CheckpointBlock(nn.Module):
    def __init__(self,module):super().__init__();self.inner=module
    def forward(self,*args,**kwargs):
        if torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self.inner,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
        return self.inner(*args,**kwargs)


class Student(nn.Module):
    def __init__(self,config):
        super().__init__()
        from worker import load_model
        from model.recogdrive_backbone import RecogDriveBackbone
        self.scientific=read(B1/'configs/repair_final_baseline.json')
        self.agent,self.identity=load_model(self.scientific)
        self.backbone=RecogDriveBackbone('internvl',self.scientific['vlm'],'cuda')
        self.backbone.eval().requires_grad_(False);self.agent.eval().requires_grad_(False)
        self.backbone.model.language_model.lm_head.forward=lambda x:x
        self.frozen=[(p,p._version) for p in list(self.backbone.parameters())+list(self.agent.parameters())]
        torch.manual_seed(config['initialization_seed']);torch.cuda.manual_seed_all(config['initialization_seed'])
        self.lora_modules=[];vision_layers=len(self.backbone.model.vision_model.encoder.layers)
        language_layers=len(self.backbone.model.language_model.model.layers)
        for name,module in list(self.backbone.model.named_modules()):
            allowed=name.startswith(('vision_model.encoder.layers.','language_model.model.layers.'))
            if allowed and isinstance(module,nn.Linear):
                parent_name,leaf=name.rsplit('.',1);parent=self.backbone.model.get_submodule(parent_name)
                setattr(parent,leaf,LoRALinear(module,config['lora_rank'],config['lora_alpha']))
                self.lora_modules.append(dict(name=name,in_features=module.in_features,out_features=module.out_features))
        for prefix,count in [('vision_model.encoder.layers.',vision_layers),('language_model.model.layers.',language_layers)]:
            assert {int(v['name'][len(prefix):].split('.')[0]) for v in self.lora_modules if v['name'].startswith(prefix)}==set(range(count))
        self.backbone.model.mlp1=NativeProjectionMasters(self.backbone.model.mlp1)
        self.agent.action_head.feature_encoder.requires_grad_(True)
        trainable_original={id(p) for p in self.agent.action_head.feature_encoder.parameters()}
        self.frozen=[(p,v) for p,v in self.frozen if id(p) not in trainable_original]
        if config.get('gradient_checkpointing',True):
            for layers in [self.backbone.model.vision_model.encoder.layers,self.backbone.model.language_model.model.layers]:
                for i in range(len(layers)):layers[i]=CheckpointBlock(layers[i])
        self.eval();self.config=config
        self.module_manifest=dict(vision_layers=vision_layers,language_layers=language_layers,lora_modules=self.lora_modules,
            trainable=[dict(name=n,shape=list(p.shape),dtype=str(p.dtype),count=p.numel()) for n,p in self.named_parameters() if p.requires_grad],
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),original_frozen_parameters=sum(p.numel() for p,_ in self.frozen),
            full_projection_masters='backbone.model.mlp1.masters',full_interface_projection='agent.action_head.feature_encoder',
            action_generation_frozen_eval=True,native_BF16_VLM_FP32_action=True,inference_cache_used=False)

    def train(self,mode=True):
        super().train(mode);self.backbone.eval();self.agent.eval();return self

    def forward(self,features,token,noise_index,trajectory=True):
        from vlm_core import prepare
        inputs,_=prepare(self.backbone,features)
        hidden=self.backbone.model(**inputs).hidden_states[-1].float()
        interface=self.agent.action_head.feature_encoder(hidden)
        prediction=None
        if trajectory:prediction=self.sample(features,hidden,interface,token,noise_index)
        return dict(interface=interface,prediction=prediction,input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])

    def sample(self,features,hidden,interface,token,noise_index):
        batch={k:v.unsqueeze(0).cuda().float() for k,v in features.items() if k not in ['pixel_values','last_hidden_state']}
        batch['last_hidden_state']=hidden
        handle=self.agent.action_head.feature_encoder.register_forward_hook(lambda module,args,out:interface)
        try:
            with torch.random.fork_rng(devices=[0]),torch.autocast('cuda',enabled=False):
                value=seed(token,noise_index);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
                return self.agent.forward(batch)['pred_traj']
        finally:handle.remove()

    def verify_frozen(self):
        assert all(p._version==version and p.grad is None and not p.requires_grad for p,version in self.frozen)
        assert not self.backbone.training and not self.agent.training

    def optimizer_groups(self):
        lora=[];projections=[]
        for name,p in self.named_parameters():
            if not p.requires_grad:continue
            (lora if name.endswith(('.A','.B')) else projections).append(p)
        assert lora and projections
        return [dict(params=lora,lr=self.config['lora_learning_rate'],name='lora'),
                dict(params=projections,lr=self.config['projection_learning_rate'],name='projections')]


def losses(outputs,target,noise_index):
    h=outputs['interface'][0];teacher=torch.as_tensor(target['teacher_interface'],device=h.device,dtype=torch.float32)
    assert h.shape==teacher.shape
    lh=F.mse_loss(h.float(),teacher)
    lt=None
    if outputs['prediction'] is not None:
        prediction=outputs['prediction'][0];trajectory=torch.as_tensor(target['teacher_trajectories'][noise_index],device=h.device,dtype=torch.float32)
        scales=prediction.new_tensor([40.,10.])
        lt=F.smooth_l1_loss(prediction[:,:2]/scales,trajectory[:,:2]/scales)+.1*(1-torch.cos(prediction[:,2]-trajectory[:,2])).mean()
    return lh,lt

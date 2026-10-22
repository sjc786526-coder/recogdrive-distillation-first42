"""Independent final intervention engine v4, including full-space projection nulls with matched-operator nulls and QDA gradients with exact prefix continuation.

Model weights and the frozen collection runtime are never modified. CPU transfers
are batched by the caller; every edit is clipped after native BF16 rounding.
"""
from common import *
import numpy as np
from vlm_core import Runtime,VOUT,VISION_GROUPS,LANGUAGE_GROUPS,prepare,vision_masks,bridge_masks,language_masks

class Engine(Runtime):
    def capture(self,name,h):
        if self.keep:self.prefix[name]=h.detach().clone()

    def forward_native(self,features):
        import torch
        inputs,_=prepare(self.backbone,features);patches=len(features['pixel_values'])
        self.vm=vision_masks(patches,'cuda');self.bm=bridge_masks(patches,'cuda');self.lm=language_masks(self.backbone,inputs['input_ids'],inputs['attention_mask'],patches)
        self.prefix={};self.kwargs={};self.collect=True;self.keep=True
        with torch.no_grad():hidden=self.model(**inputs).hidden_states[-1].float()
        self.collect=False;self.keep=False;self.inputs=inputs;self.hidden=hidden
        self.prefix['interface_pre']=hidden
        self.prefix['interface_post']=self.agent.action_head.feature_encoder(hidden).detach()
        return hidden

    def suffix(self,block,value):
        import torch
        self.collect=False;self.keep=False;h=value
        if block=='interface_post':return self.hidden,h
        if block=='interface_pre':return h,None
        if block.startswith('vision'):
            start=0 if block=='vision_embedding' else int(block.split('_')[1])+1
            layers=self.model.vision_model.encoder.layers
            for layer in layers[start:]:h=layer(h)
            assert self.model.select_layer==-1,'Explicit handling required for non-final visual selection'
            h=h[:,1:,:];side=int(h.shape[1]**.5);h=h.reshape(h.shape[0],side,side,-1)
            h=self.model.pixel_shuffle(h,scale_factor=self.model.downsample_ratio);h=h.reshape(h.shape[0],-1,h.shape[-1]);block='bridge_pre'
        if block.startswith('bridge'):
            start=0 if block=='bridge_pre' else int(block.split('_')[1])+1
            for layer in self.model.mlp1[start:]:h=layer(h)
            merged=self.prefix['language_input'].clone();selected=self.inputs['input_ids'][0]==self.model.img_context_token_id
            merged[0,selected]=merged[0,selected]*0.+h.reshape(-1,h.shape[-1]);h=merged;block='language_input'
        if block=='language_norm':return h.float(),None
        assert block=='language_input' or block.startswith('language_'),block
        start=0 if block=='language_input' else int(block.split('_')[1])+1
        for i in range(start,len(self.model.language_model.model.layers)):
            h=self.model.language_model.model.layers[i](h,**self.kwargs[f'language_{i:02d}'])[0]
        h=self.model.language_model.model.norm(h)
        return h.float(),None

    def mask(self,block,group):
        if block.startswith('vision'):matrix,valid=self.vm;groups=VISION_GROUPS
        elif block.startswith('bridge'):matrix,valid=self.bm;groups=VISION_GROUPS
        else:matrix,valid=self.lm;groups=LANGUAGE_GROUPS
        i=groups.index(group);return matrix[i]>0,bool(valid[i])

    def mask(self,block,group):
        if group in ['command_v2','requirements_v2']:
            from vlm_semantic import semantic_masks
            if getattr(self,'semantic_inputs',None) is not self.inputs:
                self.semantic_matrix,_=semantic_masks(self.backbone.tokenizer,self.inputs['input_ids'],self.inputs['attention_mask']);self.semantic_inputs=self.inputs
            return self.semantic_matrix[['command_v2','requirements_v2'].index(group)]>0,True
        if block.startswith('vision'):matrix,valid=self.vm;groups=VISION_GROUPS
        elif block.startswith('bridge'):matrix,valid=self.bm;groups=VISION_GROUPS
        else:matrix,valid=self.lm;groups=LANGUAGE_GROUPS
        i=groups.index(group);return matrix[i]>0,bool(valid[i])

    def edited(self,arm,fit):
        import torch
        block=arm['block'];h=self.prefix[block];mask,valid=self.mask(block,arm['group'])
        z=h.reshape(-1,h.shape[-1]);out=z.clone();zero=z.new_zeros(4,dtype=torch.float32)
        if not valid:return h,zero
        selected=z[mask].float();v=selected.mean(0);b=fit['basis'];c=fit['center'];p=(v-c)@b;mh=fit['mh'];mf=fit['mf'];method=arm['method']
        if method in ['mean','reverse','haar']:delta=mh-mf
        elif method=='contract':delta=mh-p
        elif method=='remove':
            w=mf-mh;w=w/w.norm().clamp(min=1e-12);delta=-torch.relu((p-mh)@w)*w
        elif method=='ot':delta=mh+(p-mf)@fit['transport'].T-p
        elif method=='quality':delta=fit['quality_direction']
        elif method=='local':
            prototypes=fit['prototypes'];dist=((prototypes-p)/fit['scale']).square().mean(1);value,ix=torch.topk(dist,min(8,len(dist)),largest=False)
            delta=(torch.softmax(-value,0)[:,None]*prototypes[ix]).sum(0)-p
        elif method=='project':delta=-p
        elif method=='soft_clip':delta=torch.clamp(p-mh,min=-2*fit['scale'],max=2*fit['scale'])-(p-mh)
        elif method=='qda_gradient':
            delta=fit['cov_inv_f']@(p-mf)-fit['cov_inv_s']@(p-mh)
            delta=delta/delta.norm().clamp(min=1e-12)*fit['scale'].norm()
        elif method=='support_projection':delta=None
        else:raise ValueError(method)
        if method=='reverse':delta=-delta
        if method=='support_projection':
            direction=c+b@p-v
            if arm.get('control_kind')=='haar_rotation':direction=fit['full_haar']@direction
        else:direction=fit['haar']@delta if (method=='haar' or arm.get('control_kind')=='haar_rotation') else b@delta
        if arm.get('control_kind')=='reverse':direction=-direction
        gate=v.new_ones(())
        if arm.get('gate','none')!='none':
            risk=((p-fit['benefit_center'])/fit['benefit_scale'])@fit['benefit_weights']+fit['benefit_intercept'] if arm['gate']=='benefit' else (v-fit['risk_center'])@fit['risk_direction']
            threshold=float(arm.get('gate_threshold',0. if arm['gate']=='benefit' else 1.))
            gate=torch.sigmoid((risk-threshold)*4) if arm['gate']=='soft' else (risk>threshold).float()
        change=direction*float(arm['alpha'])*gate
        # Broadcast shift realizes exactly the proposed change of the pooled mean.
        limit=float(arm['cap'])*selected.norm();norm=change.norm()*np.sqrt(len(selected));factor=torch.clamp(limit/norm.clamp(min=1e-12),max=1.)
        change=change*factor;trial=(selected+change).to(h.dtype)
        # BF16 rounding can exceed a proposed FP32 cap. Recompute and shrink, then
        # fall back to an unchanged group if finite iterations cannot meet it.
        for _ in range(3):
            actual=trial.float()-selected;actual_norm=actual.norm();shrink=torch.clamp(limit/actual_norm.clamp(min=1e-12)*.999,max=1.)
            change=change*shrink;trial=(selected+change).to(h.dtype)
        actual=trial.float()-selected;actual_norm=actual.norm();accept=actual_norm<=limit+1e-7
        trial=torch.where(accept,trial,z[mask]);actual=trial.float()-selected
        out[mask]=trial;stats=torch.stack([actual.norm(),selected.norm(),gate,factor])
        return out.reshape_as(h),stats

    def combined(self,parts,fit_lookup):
        import torch
        handles=[];stats=[];post_parts=[]
        def apply(part,value):
            block=part['block'];original=self.prefix.get(block);self.prefix[block]=value
            try:changed,stat=self.edited(part,fit_lookup(part));stats.append(stat);return changed
            finally:self.prefix[block]=original
        try:
            for original in parts:
                part=dict(original,alpha=original['alpha']/np.sqrt(len(parts)),cap=original['cap']/np.sqrt(len(parts)))
                block=part['block']
                if block=='interface_post':post_parts.append(part);continue
                if block in ['bridge_pre','language_input']:
                    module=self.model.mlp1[0] if block=='bridge_pre' else self.model.language_model.model.layers[0]
                    def pre(module,args,part=part):return (apply(part,args[0]),*args[1:])
                    handles.append(module.register_forward_pre_hook(pre))
                    continue
                if block=='interface_pre':block='language_norm'
                if block=='vision_embedding':module=self.model.vision_model.embeddings
                elif block.startswith('vision_'):module=self.model.vision_model.encoder.layers[int(block.split('_')[1])]
                elif block.startswith('bridge_'):module=self.model.mlp1[int(block.split('_')[1])]
                elif block=='language_norm':module=self.model.language_model.model.norm
                else:module=self.model.language_model.model.layers[int(block.split('_')[1])]
                def hook(module,args,output,part=part):
                    h=output[0] if isinstance(output,tuple) else output;changed=apply(part,h)
                    return (changed,*output[1:]) if isinstance(output,tuple) else changed
                handles.append(module.register_forward_hook(hook))
            hidden=self.model(**self.inputs).hidden_states[-1].float()
        finally:
            for handle in handles:handle.remove()
        post=self.agent.action_head.feature_encoder(hidden) if post_parts else None
        for part in post_parts:post=apply(part,post)
        zz=torch.stack(stats);relative=(zz[:,0]/zz[:,1].clamp(min=1e-12)).max()
        # Encode maximum site-relative change in the common 4-field stats format.
        metrics=torch.stack([relative,relative.new_ones(()),zz[:,2].mean(),zz[:,3].min()])
        return hidden,post,metrics

    def trajectory(self,features,hidden,value,interface=None):
        import torch
        from worker import predict
        handle=None
        if interface is not None:
            handle=self.agent.action_head.feature_encoder.register_forward_hook(lambda module,args,output:interface)
        try:
            # Keep the representation on device, avoiding a ~17MB D2H/H2D round trip.
            batch={k:v.unsqueeze(0).cuda().float() for k,v in features.items() if k!='pixel_values' and k!='last_hidden_state'}
            batch['last_hidden_state']=hidden
            reseed(value)
            with torch.no_grad(),torch.autocast('cuda',enabled=False):return self.agent.forward(batch)['pred_traj'].squeeze(0)
        finally:
            if handle is not None:handle.remove()

    def verify_engine(self):
        self.verify_weights();assert all(p.grad is None for p in self.agent.parameters())

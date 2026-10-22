"""Frozen VLM and action-interface adapters, broad pooled readouts and reusable prefixes.

No shared code/weights are changed. Token masks depend only on actual model inputs.
The unused vocabulary head can be bypassed after exact engineering verification.
"""
from common import *
import functools
import pickle
import numpy as np

VOUT=ROOT/'vlm_repair_v1'
VISION_GROUPS=['patch_mean','cls_mean','thumbnail_mean']+[f'tile_{j:02d}' for j in range(13)]+[f'grid_{j:02d}' for j in range(16)]
LANGUAGE_GROUPS=['all_mean','valid_mean','image_mean','text_mean','padding_mean','last_token','last8','last32','system','history','command','requirements']+[f'image_tile_{j:02d}' for j in range(13)]+[f'image_grid_{j:02d}' for j in range(16)]

@functools.lru_cache(maxsize=2)
def raw_log(path):
    compatibility()
    return retry(lambda:pickle.loads(Path(path).read_bytes()))

def raw_features(row):
    from navsim.common.dataclasses import AgentInput,SensorConfig
    from model.recogdrive_features import ReCogDriveFeatureBuilder
    data=raw_log(row['raw_path']);i=int(row['frame_index']);assert data[i]['token']==row['token']
    sensors=SensorConfig(cam_f0=[3],cam_l0=[],cam_l1=[],cam_l2=[],cam_r0=[],cam_r1=[],cam_r2=[],cam_b0=[],lidar_pc=[])
    frames=data[i-3:i+1];assert len(frames)==4
    inp=AgentInput.from_scene_dict_list(frames,Path(row['sensor_root']),4,sensors)
    return ReCogDriveFeatureBuilder(cache_hidden_state=False).compute_features(inp)

def question(features):
    from model.recogdrive_features import format_number
    history=features['history_trajectory'];command=features['high_command_one_hot']
    choices=['turn left','go straight','turn right','stopped']
    name=next((choices[i] for i,v in enumerate(command) if v==1),'unknown')
    text=' '.join([f'   - t-{3-i}: ({format_number(history[i,0].item())}, {format_number(history[i,1].item())}, {format_number(history[i,2].item())})' for i in range(4)])
    prompt=f"<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n1. Visual perception from front camera view\n2. Historical motion context (last 4 timesteps):{text}\n3. Active navigation command: [{name.upper()}]"
    requirements="\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places"
    return prompt+requirements

def subsequence(values,pattern,start=0):
    for i in range(start,len(values)-len(pattern)+1):
        if values[i:i+len(pattern)]==pattern:return i
    return None

def mask_matrix(masks,length,device):
    import torch
    out=torch.zeros((len(masks),length),dtype=torch.float32,device=device);valid=[]
    for j,mask in enumerate(masks):
        ix=np.flatnonzero(mask);valid.append(bool(len(ix)))
        if len(ix):out[j,torch.as_tensor(ix,device=device)]=1/len(ix)
    return out,np.asarray(valid)

def vision_masks(patches,device):
    total=patches*1025;indices=np.arange(total);tile=indices//1025;position=indices%1025
    masks=[position>0,position==0,(tile==patches-1)&(position>0)]
    masks += [(tile==j)&(position>0) for j in range(13)]
    y=(position-1)//32;x=(position-1)%32
    masks += [(position>0)&(y//8==j//4)&(x//8==j%4) for j in range(16)]
    return mask_matrix(masks,total,device)

def bridge_masks(patches,device):
    total=patches*256;indices=np.arange(total);tile=indices//256;position=indices%256
    masks=[np.ones(total,dtype=bool),np.zeros(total,dtype=bool),tile==patches-1]
    masks += [tile==j for j in range(13)]
    y=position//16;x=position%16
    masks += [(y//4==j//4)&(x//4==j%4) for j in range(16)]
    return mask_matrix(masks,total,device)

def language_masks(backbone,input_ids,attention,patches):
    values=input_ids[0].cpu().tolist();att=attention[0].bool().cpu().numpy();image=np.asarray(values)==backbone.img_context_token_id
    n=len(values);indices=np.arange(n);valid_indices=indices[att]
    masks=[np.ones(n,dtype=bool),att,image,att&~image,~att,indices==n-1,indices>=n-8,indices>=n-32]
    def locate(text):return subsequence(values,backbone.tokenizer.encode(text,add_special_tokens=False))
    history=locate('2. Historical motion context');command=locate('3. Active navigation command');requirements=locate('Output requirements:')
    image_indices=np.flatnonzero(image);first_image=int(image_indices[0]) if len(image_indices) else n
    masks.append(att&(indices<first_image))
    for begin,end in [(history,command),(command,requirements),(requirements,n)]:
        masks.append((indices>=begin)&(indices<end)&att if begin is not None and end is not None else np.zeros(n,dtype=bool))
    assert int(image.sum())==patches*256
    for j in range(13):
        mask=np.zeros(n,dtype=bool);mask[image_indices[j*256:(j+1)*256]]=True;masks.append(mask)
    positions=np.arange(len(image_indices))%256;y=positions//16;x=positions%16
    for j in range(16):
        mask=np.zeros(n,dtype=bool);mask[image_indices[(y//4==j//4)&(x//4==j%4)]]=True;masks.append(mask)
    return mask_matrix(masks,n,input_ids.device)

def prepare(backbone,features):
    import torch
    from model.recogdrive_backbone import system_message,IMG_START_TOKEN,IMG_CONTEXT_TOKEN,IMG_END_TOKEN
    from model.utils.conversation import get_conv_template
    pixels=features['pixel_values'];patches=len(pixels);q=question(features)
    template=get_conv_template('internvl2_5');template.system_message=system_message
    template.append_message(template.roles[0],q);template.append_message(template.roles[1],None)
    query=template.get_prompt().replace('<image>',IMG_START_TOKEN+IMG_CONTEXT_TOKEN*256*patches+IMG_END_TOKEN,1)
    backbone.tokenizer.padding_side='left';encoded=backbone.tokenizer([query],return_tensors='pt',padding='max_length',max_length=2800)
    ids=encoded['input_ids'].cuda();attention=encoded['attention_mask'].cuda();position=attention.long().cumsum(-1)-1;position.masked_fill_(attention==0,1)
    return dict(pixel_values=pixels.cuda().to(torch.bfloat16),input_ids=ids,attention_mask=attention,position_ids=position,
                image_flags=torch.ones(patches,dtype=torch.long),use_cache=False,output_hidden_states=True,return_dict=True),q

class Runtime:
    def __init__(self,fast=True):
        import torch
        from worker import load_model
        from model.recogdrive_backbone import RecogDriveBackbone
        self.cfg=read(ROOT/'configs/repair_final_baseline.json');self.agent,self.planner_runtime=load_model(self.cfg)
        self.backbone=RecogDriveBackbone('internvl',self.cfg['vlm'],'cuda');self.backbone.eval().requires_grad_(False)
        self.model=self.backbone.model;self.fast=fast;self.original_head=self.model.language_model.lm_head.forward
        if fast:self.model.language_model.lm_head.forward=lambda x:x
        self.versions={k:p._version for k,p in self.backbone.named_parameters()}
        self.handles=[];self.collect=False;self.keep=False;self.blocks={};self.prefix={};self.kwargs={}
        modules=[('vision_embedding',self.model.vision_model.embeddings)]+[(f'vision_{i:02d}',m) for i,m in enumerate(self.model.vision_model.encoder.layers)]
        modules += [('bridge_pre',self.model.mlp1[0])]+[(f'bridge_{i}',m) for i,m in enumerate(self.model.mlp1)]
        # bridge_pre is captured with a prehook; output hooks cover every true module.
        for name,m in modules:
            if name=='bridge_pre':self.handles.append(m.register_forward_pre_hook(self.prehook(name)))
            else:self.handles.append(m.register_forward_hook(self.hook(name)))
        for i,m in enumerate(self.model.language_model.model.layers):
            name=f'language_{i:02d}';self.handles.append(m.register_forward_hook(self.hook(name)))
            self.handles.append(m.register_forward_pre_hook(self.kw_hook(name),with_kwargs=True))
        self.handles.append(self.model.language_model.model.norm.register_forward_hook(self.hook('language_norm')))

    def kw_hook(self,name):
        def hook(module,args,kwargs):
            if name=='language_00' and self.collect:self.capture('language_input',args[0] if args else kwargs['hidden_states'])
            if self.keep:self.kwargs[name]=kwargs
        return hook

    def prehook(self,name):
        def hook(module,inputs):
            if self.collect:self.capture(name,inputs[0])
        return hook

    def hook(self,name):
        def hook(module,inputs,output):
            if self.collect:self.capture(name,output[0] if isinstance(output,tuple) else output)
        return hook

    def capture(self,name,h):
        if name.startswith('vision'):matrix,valid=self.vm
        elif name.startswith('bridge'):matrix,valid=self.bm
        else:matrix,valid=self.lm
        values=matrix@h.reshape(-1,h.shape[-1]).float()
        values[~valid]=float('nan');self.blocks[name]=values
        if self.keep:self.prefix[name]=h.detach().clone()

    def forward(self,features,keep=False):
        import torch
        inputs,q=prepare(self.backbone,features);patches=len(features['pixel_values'])
        self.vm=vision_masks(patches,'cuda');self.bm=bridge_masks(patches,'cuda');self.lm=language_masks(self.backbone,inputs['input_ids'],inputs['attention_mask'],patches)
        self.blocks={};self.prefix={};self.kwargs={};self.collect=True;self.keep=keep
        with torch.no_grad():out=self.model(**inputs);hidden=out.hidden_states[-1].float()
        self.collect=False;self.keep=False
        self.inputs=inputs
        self.blocks['interface_pre']=self.lm[0]@hidden[0]
        projected=self.agent.action_head.feature_encoder(hidden)
        self.blocks['interface_post']=self.lm[0]@projected[0]
        for name in ['interface_pre','interface_post']:self.blocks[name][~self.lm[1]]=float('nan')
        return hidden,{k:v.detach().cpu().numpy() for k,v in self.blocks.items()},dict(patches=patches,tokens=hidden.shape[1],vision_groups=VISION_GROUPS,language_groups=LANGUAGE_GROUPS)

    def verify_weights(self):
        assert self.versions=={k:p._version for k,p in self.backbone.named_parameters()}
        assert all(p.grad is None for p in self.backbone.parameters())

def smoke(output,anchors=2,return_runtime=False):
    import torch
    from worker import load_features,predict
    from repair_engine import baseline_row
    from repair_score import score_group
    runtime=Runtime(fast=True);checks=[]
    for row in read(ROOT/'repair_v1/anchors.json')[:anchors]:
        cached,_,meta=load_features(row,runtime.cfg);raw=raw_features(row)
        for key in ['history_trajectory','high_command_one_hot','status_feature']:assert torch.equal(raw[key],cached[key]),key
        began=time.monotonic();hidden,blocks,layout=runtime.forward(raw,keep=True);torch.cuda.synchronize();seconds=time.monotonic()-began
        assert torch.equal(hidden[0].cpu(),cached['last_hidden_state']),(row['token'],float((hidden[0].cpu()-cached['last_hidden_state']).abs().max()))
        # Validate bypassing only the unused output vocabulary head, against original forward.
        runtime.model.language_model.lm_head.forward=runtime.original_head
        with torch.no_grad():original=runtime.model(**runtime.inputs).hidden_states[-1].float()
        assert torch.equal(hidden,original);runtime.model.language_model.lm_head.forward=lambda x:x
        altered=dict(cached,last_hidden_state=hidden[0].cpu());prediction=predict(runtime.agent,altered,seed(row['token']));reference=baseline_row(row)
        assert np.array_equal(prediction,reference['trajectory'])
        score=score_group(dict(metric_cache=row['metric_cache'],metric_sha256=reference['metric_sha256'],scoring=runtime.cfg['scoring'],records=[dict(trajectory=prediction.tolist(),arm='baseline')]))[0]['score']
        assert score==reference['score'];runtime.verify_weights()
        checks.append(dict(token=row['token'],seconds=seconds,layout=layout,block_shapes={k:list(v.shape) for k,v in blocks.items()},cache_hidden_exact=True,unused_head_bypass_exact=True,trajectory_exact=True,score_exact=True))
    atomic(output,dict(status='PASS',time=stamp(),checks=checks,vlm_weights_unchanged=True,planner=runtime.planner_runtime))
    print(json.dumps(dict(status='PASS',checks=len(checks),seconds=[c['seconds'] for c in checks])),flush=True)
    if return_runtime:return runtime

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--smoke',required=True);args=p.parse_args();setup('vlm_repair/smoke');compatibility();smoke(ROOT/args.smoke)

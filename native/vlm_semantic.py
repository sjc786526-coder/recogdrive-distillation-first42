"""Correct two text-span pools; the colon was merged with the following newline by BPE."""
from common import *
import numpy as np
from vlm_core import Runtime,VOUT,prepare,subsequence,mask_matrix
SEM=ROOT/'vlm_semantic_v2'
GROUPS=['command_v2','requirements_v2']

def semantic_masks(tokenizer,ids,attention):
    values=ids[0].cpu().tolist();valid=attention[0].bool().cpu().numpy();n=len(values);positions=np.arange(n)
    starts=[]
    for anchor in ['3. Active navigation command','Output requirements']:
        at=subsequence(values,tokenizer.encode(anchor,add_special_tokens=False));assert at is not None,anchor;starts.append(at)
    command,requirements=starts;assert command<requirements<n
    masks=[valid&(positions>=command)&(positions<requirements),valid&(positions>=requirements)]
    matrix,present=mask_matrix(masks,n,ids.device);assert present.all()
    return matrix,dict(command_start=command,requirements_start=requirements,command_tokens=int(masks[0].sum()),requirements_tokens=int(masks[1].sum()),requirements_includes_trailing_template=True)

class SemanticRuntime(Runtime):
    def capture(self,name,h):
        if name.startswith('language'):self.blocks[name]=self.semantic_matrix@h.reshape(-1,h.shape[-1]).float()

    def forward_semantic(self,features):
        import torch
        inputs,_=prepare(self.backbone,features);self.semantic_matrix,boundaries=semantic_masks(self.backbone.tokenizer,inputs['input_ids'],inputs['attention_mask']);self.blocks={};self.collect=True;self.keep=False
        with torch.no_grad():hidden=self.model(**inputs).hidden_states[-1].float()
        self.collect=False;self.blocks['interface_pre']=self.semantic_matrix@hidden[0];projected=self.agent.action_head.feature_encoder(hidden);self.blocks['interface_post']=self.semantic_matrix@projected[0]
        arrays={k:v.detach().cpu().numpy() for k,v in self.blocks.items()};assert len(arrays)==32 and all(v.shape[0]==2 and np.isfinite(v).all() for v in arrays.values())
        return hidden,arrays,boundaries

def smoke():
    setup('vlm_semantic_v2/smoke');compatibility()
    import torch
    from vlm_core import raw_features
    from worker import load_features,predict
    from repair_engine import baseline_row
    from repair_score import score_group
    e=SemanticRuntime();checks=[]
    with torch.no_grad():
        for row in read(ROOT/'repair_v1/anchors.json')[:2]:
            raw=raw_features(row);cached,_,_=load_features(row,e.cfg);h,values,bounds=e.forward_semantic(raw);assert torch.equal(h[0].cpu(),cached['last_hidden_state'])
            before=baseline_row(row);p=predict(e.agent,dict(cached,last_hidden_state=h[0].cpu()),seed(row['token']));assert np.array_equal(p,before['trajectory'])
            score=score_group(dict(metric_cache=row['metric_cache'],metric_sha256=before['metric_sha256'],scoring=e.cfg['scoring'],records=[dict(arm='baseline',trajectory=p.tolist())]))[0]['score'];assert score==before['score']
            checks.append(dict(token=row['token'],boundaries=bounds,shapes={k:list(v.shape) for k,v in values.items()},native_hidden_exact=True,trajectory_score_exact=True))
    e.verify_weights();atomic(SEM/'ENGINEERING_PASS.json',dict(status='PASS',time=stamp(),checks=checks,source_issue='BPE colon+newline token means the colon-terminated substring was absent; anchor without colon matches actual full token stream',old_command_requirements_invalid=True,all_other_original_pools_unchanged=True))
    print('SEMANTIC_MASK_NATIVE_AND_SCORING_PASS',flush=True)
if __name__=='__main__':smoke()

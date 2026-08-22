from pathlib import Path
p=Path('/data/models/Kimi-K3/modeling_kimi_linear.py')
s=p.read_text()
old="""_K3_PENDING=[]; _K3_PARTS=[]
def _k3_ev():
 import os
 if not os.environ.get('K3_EVENT_TIMING'): return None
 e=torch.cuda.Event(enable_timing=True); e.record(); return e
def _k3_flush(a,b,tokens):
 import os,json
 if a is None: return
 _K3_PENDING.append((a,b,tokens))
 if len(_K3_PENDING)<16: return
 torch.cuda.synchronize(); rank=os.environ.get('RANK','?')
 for st,en,tok in _K3_PENDING:
  total=st.elapsed_time(en); attn=sum(x.elapsed_time(y) for n,x,y in _K3_PARTS if n=='attention'); moe=sum(x.elapsed_time(y) for n,x,y in _K3_PARTS if n=='moe')
  print('K3_EVENT_TIMING_V1 '+json.dumps({'rank':rank,'tokens':int(tok),'model_ms':total,'attention_ms':attn,'moe_ms':moe,'other_ms':max(0.,total-attn-moe)},separators=(',',':')),flush=True)
 _K3_PENDING.clear(); _K3_PARTS.clear()
"""
new="""_K3_PENDING=[]; _K3_PARTS=[]
def _k3_ev():
 import os
 if not os.environ.get('K3_EVENT_TIMING'): return None
 e=torch.cuda.Event(enable_timing=True); e.record(); return e
def _k3_flush(a,b,tokens,parts):
 import os,json
 if a is None: return
 _K3_PENDING.append((a,b,tokens,parts))
 if len(_K3_PENDING)<16: return
 torch.cuda.synchronize(); rank=os.environ.get('RANK','?')
 for st,en,tok,ps in _K3_PENDING:
  total=st.elapsed_time(en); attn=sum(x.elapsed_time(y) for n,x,y in ps if n=='attention'); moe=sum(x.elapsed_time(y) for n,x,y in ps if n=='moe')
  print('K3_EVENT_TIMING_V1 '+json.dumps({'rank':rank,'tokens':int(tok),'model_ms':total,'attention_ms':attn,'moe_ms':moe,'other_ms':max(0.,total-attn-moe)},separators=(',',':')),flush=True)
 _K3_PENDING.clear()
"""
assert old in s
s=s.replace(old,new,1)
oldcall="_k3_flush(_k3_model_start,_k3_model_end,inputs_embeds.shape[1])"
assert oldcall in s
s=s.replace(oldcall,"_k3_flush(_k3_model_start,_k3_model_end,inputs_embeds.shape[1],list(_K3_PARTS))\n        _K3_PARTS.clear()",1)
p.write_text(s); print('repaired')

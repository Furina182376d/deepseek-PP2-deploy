from pathlib import Path
p=Path('/data/models/Kimi-K3/modeling_kimi_linear.py')
s=p.read_text()
old=""" if not os.environ.get('K3_EVENT_TIMING'): return None
 e=torch.cuda.Event(enable_timing=True); e.record(); return e
"""
new=""" if not torch.cuda.is_available(): return None
 e=torch.cuda.Event(enable_timing=True); e.record(); return e
"""
assert old in s
p.write_text(s.replace(old,new,1))
print('forced')

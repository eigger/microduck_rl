import sys
sys.path.insert(0, 'scripts')
from infer_policy import load_bam_model, BAM_KP_FW

bm = load_bam_model(BAM_KP_FW, 7.4, None)
print("BAM model attributes:")
for k in dir(bm):
    if not k.startswith('_'):
        print(f"  {k}: {getattr(bm, k)}")

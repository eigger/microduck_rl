import sys
sys.path.insert(0, 'scratch')
from eval_run21 import evaluate_policy

print('=== Results for Run 22 (scratch/jump_run22.onnx) ===')
evaluate_policy('Run 22 (Additive Flight & Deadband)', 'scratch/jump_run22.onnx')

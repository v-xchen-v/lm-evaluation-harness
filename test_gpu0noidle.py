"""
https://github.com/ultralytics/yolov5/discussions/6649

larger batch_size could bring shorter inference time per token. 
1. good gain vs batch-size 1
2. not linear gain
"""
import torch
import time
from transformers import AutoModelForCausalLM

# model
hf_model_id="gpt2"
# hf_model_id="gpt2-medium"
# hf_model_id="gpt2-large"
# hf_model_id="gpt2-xl"
device_map={"":0}
# device_map="auto"

def batchsize_timer(model):
    batch_size = 1
    max_length = 200 #model.config.n_positions the max_length of model input
    device = model.device if model.hf_device_map is None else list(set(model.hf_device_map.values()))
    try:
        while True:
            # dummy input
            test_batch = torch.ones((batch_size, max_length), device=model.device).long() # long() is int64 takes 4 bytes the same as float32
            n_repeat = 50
            time_start = time.perf_counter()
            for i in range(0, n_repeat):
                model(test_batch)
            time_stop = time.perf_counter()
    
            print(f"inferece {hf_model_id} batch_size: {batch_size} on {device}")
            print(f"Inference a sentence in {(time_stop - time_start)/n_repeat:0.4f} seconds")
            print(f"Inference a sentence in {((time_stop - time_start)*1000/n_repeat)/(max_length*batch_size):0.4f} ms/token")
            # torch.cuda.empty_cache()
            batch_size *= 2
    except RuntimeError:# torch.cuda.OutOfMemoryError:
        print(f"OOM on {device}")

model = AutoModelForCausalLM.from_pretrained(hf_model_id, device_map=device_map)
while True:
    print(model.hf_device_map)
    batchsize_timer(model)
"""
|    | gpt2-xl | gpt2-large | gpt2-medium | gpt2   |                 |
|----|---------|------------|-------------|--------|-----------------|
| 1  | 0.5981  | 0.4573     | 0.3244      | 0.2337 | times(ms/token) |
| 2  | 0.3770  | 0.2202     | 0.1258      | 0.0651 |                 |
| 4  | 0.3440  | 0.1736     | 0.0894      | 0.0399 |                 |
| 8  | -       | 0.1682     | 0.0778      | 0.0317 |                 |
| 16 | -       | -          | 0.0778      | 0.0283 |                 |
| 32 | -       | -          | -           | 0.0283 |                 |
"""


import subprocess
from pathlib import Path

seq = ['04', '05', '07', '10', '09']
result_path = '/home/yyjau/Documents/DeepVO-pytorch/result'
cvt_path = './result_deepvo'

Path(cvt_path).mkdir(parents=True, exist_ok=True)

for s in seq:
    print(f"seq: {s}")
    command = f'python qua2mat.py {result_path}/out_{s}.txt {cvt_path}/{s}_pred.txt'
    subprocess.run(f"{command}", shell=True, check=True)
    command = f"python evaluation.py --result_dir={cvt_path} --eva_seqs={s}_pred"
    subprocess.run(f"{command}", shell=True, check=True)



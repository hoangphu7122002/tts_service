import time
import torch
import sys
import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='0',
                    required=False)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

argslist = list(sys.argv)[1:]
num_gpus = torch.cuda.device_count()
argslist.append('--n_gpus={}'.format(num_gpus))
workers = []
job_id = time.strftime("%Y_%m_%d-%H%M%S")
argslist.append("--group_name=group_{}".format(job_id))

for i in range(num_gpus):
    argslist.append('--rank={}'.format(i))
    stdout = None if i == 0 else open("gpu_logs/{}_GPU_{}.log".format(job_id, i),
                                      "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)]+argslist, stdout=stdout)
    workers.append(p)
    argslist = argslist[:-1]

for p in workers:
    try:
        p.wait()
    except KeyboardInterrupt:
        p.wait()
        sys.exit(0)

import numpy as np
TARGET=["s-nll","a-nll","s-kl","a-kl","i-l","sa-l","aa-l"]
header = ",".join(["epoch","step"]+TARGET)+'\n'
fout=open("vrbot-metrics.csv","w")
fout.write(header)
with open("meddg_train.log") as fin:
    for i,line in enumerate(fin):
        line = line.rstrip()
        tid = [True if x in line  else False  for x in TARGET]
        if np.all(tid):
           #TARGETList.append(line)
           snll_idx = line.find("s-nll")
           epoch_ids = line.find("EPOCH")
           epoch_step = line[epoch_ids:].split(" ")
           epoch,step = epoch_step[1],epoch_step[3]
           metric = line[snll_idx:]
           ml = metric.split(" ")
           me = ",".join([epoch,step]+[ml[i+1] for i in range(0,len(ml),2)])+"\n"
           fout.write(me)
fout.close()

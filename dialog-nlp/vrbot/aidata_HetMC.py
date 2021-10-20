import os
import json
import numpy as np
import pandas as pd
import argparse

############################
parser = argparse.ArgumentParser()
parser.add_argument("--data",type=str,default=None,help="aidata csv file")
parser.add_argument("--rate",type=float,default=0.8,help="train rate")
parser.add_argument("--outdir",type=str,default="dialog",help="output path to save result")
args = parser.parse_args()


infile=args.data
rate=args.rate
outdir=args.outdir

if not os.path.exists(outdir):
   os.makedirs(outdir)

############################
print("读取数据，得到每轮对话")
check_talk_round=[]
talk_message=[]
###  如果没有error='ignore'，在6089行会报错
with open(infile,encoding="gbk",errors='ignore') as fin:
    for i,line in enumerate(fin):
        if i==0:
            continue
        #if i>10000:
        #   break
        line=line.rstrip()
        line_list=line.split(",")
        check_talk_round.append(line_list[0])
        talk_message.append(line_list[2])
fin.close()

idx=[]   
i=0
for v in check_talk_round:
    if i+1>=len(check_talk_round):
        break
    vadd=check_talk_round[i+1]
    if v!=vadd:
        idx.append(i+1)
    i+=1
idx=[0]+idx

talk_message_list=[]
check_talk_round_list=[]
for i,_ in enumerate(idx):
    if i<(len(idx)-1):
        value=talk_message[idx[i]:idx[i+1]]
        tround=check_talk_round[idx[i]:idx[i+1]]
    else:
        value=talk_message[idx[i]:]
        tround=check_talk_round[idx[i]:]
    
    talk_message_list.append(value)
    check_talk_round_list.append(tround)


#########################
print("去掉需要看图片的对话")    
#### 去掉需要看图片的对话
has_picture_talk=["示意图","参考图片","展示图片","图片展示","请看示意图"]
remove_picture_index=[]
for i in range(len(talk_message_list)):
    talk_message=talk_message_list[i]
    talk_round=check_talk_round_list[i]
    flag=False
    for v in talk_message:
        for p in has_picture_talk:
            if p in v:
                flag=True
                break
        if flag:
            break
    if flag:
        remove_picture_index.append(i)
        continue
        

clean_talk_message_list=[talk_message_list[i] for i in range(len(talk_message_list)) if i not in remove_picture_index]
clean_check_talk_round_list=[check_talk_round_list[i] for i in range(len(check_talk_round_list)) if i not in remove_picture_index]


##########################
print("split into train and test")
n_train=int(len(clean_talk_message_list)*rate)
train_idx=np.random.choice(range(len(clean_talk_message_list)),n_train)
test_idx=[i for i in range(len(clean_talk_message_list)) if i not in train_idx]

test_X,test_Y=[clean_talk_message_list[i] for i in test_idx],[clean_check_talk_round_list[i] for i in test_idx]
train_X,train_Y=[clean_talk_message_list[i] for i in train_idx],[clean_check_talk_round_list[i] for i in train_idx]


##########################
prefix_dict={"病人":"P","医生":"D"}
label_dict={"病人":"1","医生":"2","Other":"0"}
pattern=u"病人|医生"

########## train
train_file=outdir+"/"+"train.dialog"
writer=open(train_file,"w")
for i,talk_message in enumerate(train_X):    
    dp=train_Y[i][0]
    if dp=="" or dp is None:
        continue
    header=f'id\t{str(i)}\tdepartment\t{dp}\tdisease\t{dp}\n'
    sum1=talk_message[0].replace("病人:","")
    sum2=talk_message[-1].replace("医生:","")
    if sum1=="" or sum2=="":
        continue
    #print(header)
    writer.write(header)
    for tm in talk_message:
        pre,content=tm.split(":",1)
        label=label_dict[pre]
        prefix=prefix_dict[pre]
        line=f'{prefix}\t{content}\t{label}\n'
        writer.write(line)
        #print(line)
    sum1=talk_message[0].replace("病人:","")
    sum2=talk_message[-1].replace("医生:","")
    writer.write(f'SUM1\t{sum1}\n')
    writer.write(f'SUM2.0\t{sum2}\n')
    writer.write("\n")
writer.close()

########### test
test_file=outdir+"/"+"test.dialog"
writer=open(test_file,"w")
for i,talk_message in enumerate(test_X):    
    dp=test_Y[i][0]
    if dp=="" or dp is None:
        continue
    header=f'id\t{str(i)}\tdepartment\t{dp}\tdisease\t{dp}\n'
    sum1=talk_message[0].replace("病人:","")
    sum2=talk_message[-1].replace("医生:","")
    if sum1=="" or sum2=="":
        continue
    #print(header)
    writer.write(header)
    for tm in talk_message:
        pre,content=tm.split(":",1)
        label=label_dict[pre]
        prefix=prefix_dict[pre]
        line=f'{prefix}\t{content}\t{label}\n'
        writer.write(line)
        #print(line)
    writer.write(f'SUM1\t{sum1}\n')
    writer.write(f'SUM2.0\t{sum2}\n')
    writer.write("\n")
writer.close()


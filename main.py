
import torch.nn as nn
from dataset.dataset import Model_Dataset
from dataset.vocab import Vocab
from module.model import Model
from torch.utils.data import DataLoader
import json
import os
import yaml
import logging
from test_and_valid import *
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.is_available()
logging.disable(30)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


config=yaml.load(open('config.yaml'))
train_path=config['train_data_dir']
valid_path=config['valid_data_dir']
test_path=config['test_data_dir']

def func(num):
    return '/gpu:'+str(num)

domains={
    "Other":0,
    "Phone":1,
    "Navi":3,
    "DeviceControl":4,
    "Media":5,
    "Weather":6,
    "General":7,
    "Robot":9,
    "manual":10,
    "WeChat":11,
    "Inquire":12,
}
'''
ques_types={
    "howmany":1,
    "howto":2,
    "what":3,
    "when":4,
    "where":5,
    "which":6,
    "why":7,
    "yn":8,
    "statement":9,
    "statement_no":10
}

ins=open('./dataset/in_vocab_10000000.txt')
natures=open('./dataset/nature_vocab_10000000.txt')
slots=open('./dataset/slot_vocab_10000.txt')
slot_intent=open('./dataset/slot_intent_vocab_10000.txt')
'''
def  to_numpy(items):
    return list(map(lambda x:x.numpy(),items))

vocabs = {'sys_intent':Vocab(),'in': Vocab(device), 'nature': Vocab(device), 'slot': Vocab(device), 'intents': Vocab(device), 'domain': Vocab(device),
          'ques_type': Vocab(device),'is_mul':Vocab(device)}
def make_dataset(path,if_vocab=True):
    inputs={}
    labels={}

    inputs_file=os.listdir(path+'input/')
    label_file=os.listdir(path+'label/')


    for file in inputs_file:
        ss=file.split('.')
        data=list(map(lambda x:x.split(' '),open(path+'input/'+file).read().split('\n')))
        inputs[ss[-1]]=data
        if if_vocab:
            if ss[-1]=='context_slots':
                for words in data:
                    vocabs['slot'].s2i_f(words)
            else:
                for words in data:
                    vocabs['nature'].s2i_f(words)

    for file in label_file:
        ss = file.split('.')
        data = list(map(lambda x: x.split(' '), open(path + 'label/' + file).read().split('\n')))
        labels[ss[-1]]=data
        if if_vocab:
            if ss[-1]=='is_multi_round':
                ss[-1]='is_mul'
            for words in data:
                vocabs[ss[-1]].s2i_f(words)
    return inputs,labels
vocabs['in'].s2i=json.loads(open('../Tranformer_Pyrorch_pre_train/vocab.json').read())

'''
max_len,x1,x2,
                 prev_x1,
                 prev_x2,
                 prev_context,
                 y1,y2,
                 domain,
                 ques_type,
                 is_mul,
                 x1_vocab,
                 x2_vocab,
                 y1_vocab,
                 y2_vocab,
'''
max_len=config['max_len']
num_layers=config['num_layers']
input_embedding_size=config['input_embedding_size']
num_heads=config['num_heads']
num_units=config['num_units']
cnn_filters=config['filters']
kernel_size=config['cnn_kernel_sizes']
learning_rate=config['learning_rate']
batch_size=config['batch_size']
devices=[i for i in range(config['devices'])]
dropout=config['dropout']
epoches=config['epoches']
beta_1=config['beta_1']
decay=config['decay']
valid_barch_size=config['valid_batch_size']
O_decayweight=config['O_decayweight']
update_weight=config['update_weight']
prob=config['prob']
pretrain_path=config['pretrain_path']
test_batch_size=config['test_batch_size']


inputs,labels=make_dataset(train_path)
v_inputs,v_labels=make_dataset(valid_path)
t_inputs,t_labels=make_dataset(test_path)
def clean_items(item):
    return item[:-1]

while not labels['domain'][-1][0]:
    label=labels['domain'][-1]
    inputs={item:clean_items(inputs[item]) for item in inputs}
    labels = {item: clean_items(labels[item]) for item in labels}

model=Model(input_embedding_size,num_layers,num_heads,max_len,num_units,len(vocabs['in'].s2i),
            len(vocabs['nature'].s2i),len(vocabs['slot'].s2i),len(vocabs['intents'].s2i),len(vocabs['domain'].s2i)+29,len(vocabs['sys_intent'].s2i),len(vocabs['ques_type'].s2i)+29,vocabs['in'].s2i['_PAD'],dropout,prob).to(device)


train_dataset=Model_Dataset(max_len,inputs['in'],inputs['nature'],inputs['prev'],inputs['prev_nature'],inputs['context_slots'],labels['slot'],labels['intents'],labels['domain'],
                      labels['sys_intent'],labels['ques_type'],labels['is_multi_round'],vocabs['in'].s2i,vocabs['nature'].s2i,vocabs['slot'].s2i,vocabs['intents'].s2i,vocabs['sys_intent'].s2i,model.encoder.token_embedding)

valid_dataset=Model_Dataset(max_len,v_inputs['in'],v_inputs['nature'],v_inputs['prev'],v_inputs['prev_nature'],v_inputs['context_slots'],v_labels['slot'],v_labels['intents'],v_labels['domain'],v_labels['sys_intent'],
                      v_labels['ques_type'],v_labels['is_multi_round'],vocabs['in'].s2i,vocabs['nature'].s2i,vocabs['slot'].s2i,vocabs['intents'].s2i,vocabs['sys_intent'].s2i,model.encoder.token_embedding)

test_dataset=Model_Dataset(max_len,inputs['in'],t_inputs['nature'],t_inputs['prev'],t_inputs['prev_nature'],t_inputs['context_slots'],t_labels['slot'],t_labels['intents'],t_labels['domain'],t_labels['sys_intent'],
                      t_labels['ques_type'],t_labels['is_multi_round'],vocabs['in'].s2i,vocabs['nature'].s2i,vocabs['slot'].s2i,vocabs['intents'].s2i,vocabs['sys_intent'].s2i,model.encoder.token_embedding)


dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
valid_data_loader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=True)
test_data_loader=DataLoader(test_dataset,batch_size=test_batch_size,shuffle=True)


loss_fn=nn.NLLLoss(ignore_index=vocabs['slot'].s2i['_PAD'],reduction='none').to(device)
optim=torch.optim.Adam(model.parameters(),lr=learning_rate)
step_decay=[i for i in range(100,10000,100)]
step_decay+=[i for i in range(10000,100000,1000)]
step_decay=torch.optim.lr_scheduler.MultiStepLR(optim,step_decay,0.99)

for vocab in vocabs:
    vocabs[vocab].get_p()

def label_smoothing(y_pred,y_true,high,padding_index):
    size=y_pred.shape[-1]

    weight=torch.zeros_like(y_pred)
    for i in range(len(y_true)):
        weight[i,:]=(1-high)/(size-1)
        weight[i][y_true[i]]=high

    mask = 1 - torch.eq(y_true, padding_index).to(torch.float)
    loss=(-y_pred*weight).sum(-1)*mask
    return loss.mean(),mask


def compute_loss(y_pred,y_true,name):

    weight=vocabs[name].p_c[y_true]
    loss=loss_fn(y_pred,y_true)
    loss=loss*torch.clamp(torch.log(weight+1),min=0.5,max=2)
    vocabs[name].p_c[y_true]=vocabs[name].p_c[y_true]*(1-update_weight)+loss.detach()*update_weight
    return loss.mean()

steps=0
for epoch in range(epoches):
    for x1,x2,prev_x1,prev_x2,prev_context,y1,y2,x_mask,prev_x_mask,y1_mask,y2_mask,domain,sys_intent,query_type,is_mul,t1,t2,O_label in dataloader:
        
        if pretrain_path and steps==1:
            model.load_state_dict(torch.load(pretrain_path))
        inputs = [x1,x2,prev_x1,prev_x2,prev_context,y1,y2,domain,query_type,is_mul,x_mask,prev_x_mask,y1_mask,y2_mask]
        inputs=list(map(lambda x:x.to(device) if x!=None else x,inputs))
        inputs+=[dropout]
        predicts=model(*inputs)
    
        labels = [ t1, O_label,domain,sys_intent,query_type, is_mul]
        labels = list(map(lambda x: x.to(device) if x!=None else x, labels))
        loss = 0



        for predict, label, type_ in zip(predicts, labels, [ 'slot', 'O','domain','sys_intent','ques_type', 'is_mul',]):
            if type_=='is_mul' or type_=='O':
                loss_1=loss_fn(predict.view(-1,predict.shape[-1]),label.view(-1))
                if type_=='O':
                    loss_1*=mask
                loss_1=loss_1.mean()
            else:
                loss_1,mask = label_smoothing(predict.view(-1,predict.shape[-1]),label.view(-1),0.8,0)

            loss += loss_1
            predict_indexes = torch.argmax(predict, axis=-1)

            if type_ == 'domain' or type_=='ques_type' or type_=='is_mul' or type_=='O':
                res_t = label[-1]
                res_p = predict_indexes[-1]
            else:
                res_t = vocabs[type_].i2s_f(label[-1])
                res_p = vocabs[type_].i2s_f(predict_indexes[-1])
            print(res_t, '\n')
            print(res_p, '\n')
            print(str(epoch)+'.'+str(steps)+':'+str(loss), '\n\n')
        steps+=1
        optim.zero_grad()
        loss.backward()
        optim.step()
        step_decay.step()
     #   if steps%1000==0:
        if steps==2:
            print(get_f1_score(valid_barch_size,valid_dataloader=valid_data_loader,model=model,start=vocabs['slot'].s2i['<s>'],device=device,vocabs=vocabs))
        if steps%1000==0:
            torch.save(model.state_dict(),'./model2.pt')














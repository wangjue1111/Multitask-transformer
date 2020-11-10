from torch.utils.data import Dataset
import numpy as np
import torch

def create_O_label(y,vocab):
    label=np.equal(np.array(y),vocab['O']).astype(np.long)
    return label

def create_seq_mask(item,vocab1,new_item=None,vocab2=None):
    mask=np.equal(item,vocab1['_PAD'])
    if new_item!=None:
        new_mask=np.not_equal(new_item,vocab2['_PAD'])
        mask=mask|new_mask
    return mask.astype(np.float32)

def create_attn_mask(size):
    mark = 1-torch.triu(torch.ones(size,size)).T
    return mark

def create_word_embedding(seq,max_len):
    seq=seq+['_PAD' for i in range(max_len-len(seq))] if max_len-len(seq)>0 else seq[:max_len]
    return seq


def to_array(items):
    return list(map(lambda x:np.array(x,dtype=np.int32),items))

def create_seq(items,vocab,max_len):
    res=[]

    for item in items:

        res.append(vocab.get(item,vocab['_UNK']))
    length=len(res)
    if len(res)<max_len:
        res+=[vocab['_PAD'] for i in range(max_len-len(res))]
    elif len(res)>max_len:
        res=res[:max_len]
    return res

class Model_Dataset(Dataset):
    def __init__(self,max_len,x1,x2,
                 prev_x1,
                 prev_x2,
                 prev_context,
                 y1,y2,
                 domain,
                 sys_intent,
                 ques_type,
                 is_mul,
                 x1_vocab,
                 x2_vocab,
                 y1_vocab,
                 y2_vocab,
                intent_vocab,
                 embedding,
                 ):
        self.max_len=max_len
        self.x1=x1
        self.x2=x2
        self.prev_x1=prev_x1
        self.prev_x2=prev_x2
        self.prev_context=prev_context
        self.y1=y1
        self.y2=y2
        self.domain=domain
        self.ques_type=ques_type
        self.x1_vocab=x1_vocab
        self.x2_vocab=x2_vocab
        self.y1_vocab=y1_vocab
        self.y2_vocab=y2_vocab
        self.is_mul=is_mul
        self.sys_intent=sys_intent
        self.intent_vocab=intent_vocab
        self.embedding=embedding


    def __len__(self):
        return len(self.x1)

    def __getitem__(self, item):

            x1=create_word_embedding(self.x1[item],self.max_len)
            x2=create_seq(self.x2[item],self.x2_vocab,self.max_len)
            y1=create_seq(['<s>']+self.y1[item],self.y1_vocab,self.max_len)
            y2=create_seq(['<s>']+self.y2[item],self.y2_vocab,self.max_len)
            t1=create_seq(self.y1[item],self.y1_vocab,self.max_len)
            t2=create_seq(self.y2[item],self.y2_vocab,self.max_len)
            prev_context=create_seq(self.prev_context[item],self.y1_vocab,self.max_len)
            prev_x1 = create_word_embedding(self.prev_x1[item],self.max_len)
            prev_x2 = create_seq(self.prev_x2[item], self.x2_vocab, self.max_len)
            x_mask=create_seq_mask(x2,self.x2_vocab)
         #   prev_context_mask=create_seq_mask(prev_context_l,self.max_len)
            prev_x_mask=create_seq_mask(prev_x2,self.x2_vocab)
         #   x_mask=tf.concat([x_mask,prev_x_mask,prev_context_mask],-1)[tf.newaxis,tf.newaxis,:]
            y1_mask=create_attn_mask(self.max_len)+create_seq_mask(y1,self.y1_vocab)
            y2_mask=create_attn_mask(self.max_len)+create_seq_mask(y2,self.y2_vocab)
            y1_mask=y1_mask.unsqueeze(0)
            y2_mask=y2_mask.unsqueeze(0)
            domain=np.array(self.domain[item],dtype=np.long)
            sys_intent=np.array(self.intent_vocab[self.sys_intent[item][0]],dtype=np.long)
            if self.ques_type[item]==['_UNK']:
                self.ques_type[item]=[1]
            ques_type=np.array(self.ques_type[item],dtype=np.long)
            is_mul=np.array(self.is_mul[item],dtype=np.long)
            O_label = create_O_label(y1, self.y1_vocab)
            return np.array(x1),np.array(x2),np.array(prev_x1),np.array(prev_x2),np.array(prev_context),np.array(y1),np.array(y2),x_mask[np.newaxis,np.newaxis,:],prev_x_mask[np.newaxis,np.newaxis,:],y1_mask.numpy(),y2_mask.numpy(),\
                   domain,sys_intent,ques_type,is_mul,np.array(t1),np.array(t2),O_label

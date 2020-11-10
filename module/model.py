from .decoder import Decoder
from .encoder import Encoder
from .embedding import Embedding
from .embedding import Positional_Embedding
from .textcnn import TextCNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,embedding_size,num_layers,num_heads,max_len,hidden,token_vocab_size,seg_vocab_size,y1_vocab_size,y2_vocab_size,domain_vocab_size,sys_intent_vocab_size,query_type_vocab_size,padding_index,dropout,prob=0.5) :
        super(Model, self).__init__()
        self.padding_index=padding_index
        self.sys_intent_vocab_size=sys_intent_vocab_size
        self.embedding_size=embedding_size
        self.domain_vocab_size=domain_vocab_size
        self.query_vocab_size=query_type_vocab_size
        self.y1_vocab_size=y1_vocab_size
        self.y2_vocab_size=y2_vocab_size
        self.encoder=Encoder(embedding_size,num_layers,num_heads,max_len,hidden,token_vocab_size,seg_vocab_size,padding_index,dropout)
        self.decoder=Decoder(embedding_size,num_layers,num_heads,max_len,hidden,y1_vocab_size,y2_vocab_size,domain_vocab_size,query_type_vocab_size,padding_index,dropout,prob=prob)
        self.textcnn=TextCNN(max_len,num_layers,3)
        self.dense_dict=nn.ModuleDict()

    def generator(self,x,vocab_size,name):
        if name not in self.dense_dict:
            self.dense_dict[name]=nn.Sequential(nn.Linear(x.shape[-1],vocab_size),nn.LogSoftmax())
        return self.dense_dict[name](x)

    def forward(self,inputs1,inputs2,prev_inputs1,prev_inputs2,prev_context,y1,y2,domain,query_type,is_mul,inputs_mask,prev_inputs_mask,y1_mask,y2_mask,dropout=None):
        encoder_outputs=self.encoder(inputs1,inputs2,inputs_mask,dropout)
        prev_encoder_outputs=self.encoder(prev_inputs1,prev_inputs2,prev_inputs_mask,dropout)
        prev_context_mask=torch.eq(prev_context,self.padding_index).unsqueeze(1).unsqueeze(1)
        src_mask=torch.cat([inputs_mask,prev_inputs_mask],-1)
        prev_context=self.decoder.y1_embedding(prev_context)
        all_encoder_outputs=torch.cat([encoder_outputs,prev_encoder_outputs],-2)
        if domain!=None:
            domain=self.generator(self.textcnn(encoder_outputs),self.domain_vocab_size,'domain')
            sys_intent=self.generator(self.decoder(y2,encoder_outputs,None,inputs_mask,'sys_intent',dropout),self.sys_intent_vocab_size,'sys_intent')
            query_type=self.generator(self.textcnn(encoder_outputs),self.query_vocab_size,'query_type')
            is_mul=self.generator(self.decoder(is_mul,torch.cat([encoder_outputs,prev_encoder_outputs],-2),None,src_mask,'is_mul',dropout),2,'is_mul;')

        y1=self.decoder(y1,all_encoder_outputs,y1_mask,src_mask,'y1',dropout)
        y1 = self.generator(y1, self.y1_vocab_size, 'y1')
   #     y2 = self.decoder(y2, encoder_outputs, y2_mask, inputs_mask, 'y2', dropout)
   #     y2 = self.generator(y2, self.y2_vocab_size, 'y2')
        O = self.generator(y1, 2, 'O')

        if domain!=None:
            return y1,O,domain,sys_intent,query_type,is_mul
        else:
            return y1,O,

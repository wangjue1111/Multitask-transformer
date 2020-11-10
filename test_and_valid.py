from sklearn.metrics import f1_score
import numpy as np
import torch

def change2O(seq1,seq2,O_index):
    indexes=torch.eq(seq2,1)
    seq1[indexes]=O_index
    return seq1

def concat(item1,item2):
    if item1==None:
        return item2
    else:
        return torch.cat([item1,item2],0)

def get_f1_score(batch_num,valid_dataloader,model,start,device,vocabs):
    predict_all=[None]*5
    valid_labels_all=[None]*5
    for i in range(batch_num):
        x1, x2, prev_x1, prev_x2, prev_context, y1, y2, x_mask, prev_x_mask, y1_mask, y2_mask, domain,sys_intent, query_type, is_mul, t1, t2,O_label=valid_dataloader.__iter__().__next__()

        valid_labels = [t1,O_label,domain,sys_intent,query_type,is_mul]
        valid_labels_all=list(map(concat,valid_labels_all,valid_labels))
        max_len=x1.shape[1]
        y0=(torch.ones(x1.shape[0],1,dtype=torch.long)*start).to(device)
        y1=(torch.ones(x1.shape[0],1,dtype=torch.long)*start).to(device)
        predicts = model(x1.to(device), x2.to(device), prev_x1.to(device), prev_x2.to(device), prev_context.to(device), y0, y1, domain, query_type, is_mul, x_mask.to(device), prev_x_mask.to(device), None,
                  None)
        y0_,O=predicts[:2]
        O=torch.argmax(O,-1)
        y0= torch.cat([y0, torch.argmax(y0_[:, -1], -1).unsqueeze(-1)], 1)
        predicts_=list(map(lambda x:x.argmax(-1),predicts[2:]))

        for i in range(1,max_len):
            y0_,O_=model(x1, x2, prev_x1, prev_x2, prev_context, y0, y1, None, None, None, x_mask, prev_x_mask, None,
                  None)
            y0,O=torch.cat([y0,torch.argmax(y0_[:,-1],-1).unsqueeze(-1)],1),torch.cat([O,torch.argmax(O_[:,-1],-1).unsqueeze(-1)],1)
        predicts_=[y0[:,1:],O]+predicts_
        predict_all=list(map(concat,predict_all,predicts_))
    predict_all=list(map(lambda x:x.reshape(-1),predict_all))
    O_index_slot=vocabs['slot'].s2i['O']
    O_index_intents=vocabs['intents'].s2i['O']
    predict_all[0]=change2O(predict_all[0],predict_all[1],O_index_slot)


    for predict, label, type_ in zip(predict_all, valid_labels_all, ['slot', 'O', 'domain','sys_intent', 'ques_type', 'is_mul', ]):
        predict,label=predict.reshape(-1),label.reshape(-1)
        if type_ not in vocabs or 'O' in vocabs[type_].s2i:
            not_pad_indexes=np.not_equal(label,0) if type_  in vocabs else not_pad_indexes
            predict=predict[not_pad_indexes]
            label=label[not_pad_indexes]
            if type_ in vocabs:
                not_O_indexes=np.not_equal(label,vocabs[type_].s2i['O'])
                O_predict = predict[not_O_indexes]
                O_label = label[not_O_indexes]
                print(type_+'_f1_score:'+str(f1(O_label,O_predict)))
            print(type_+'_f1_score:'+str(f1(label,predict)))
        else:
            print(type_+'_f1_score:'+str(f1(label.view(-1),predict.view(-1))))

def f1(y_true,y_pred):
    f1_micro = f1_score(y_true.cpu(), y_pred.cpu(), average='micro')
#    f1_macro = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='macro')
    return f1_micro

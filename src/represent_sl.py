# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 19:54:17 2021

@author: luol2
"""



import os, sys
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer
           

class Hugface_RepresentationLayer(object):
    
    
    def __init__(self, tokenizer_name_or_path, label_file,lowercase=True):
        

        #load vocab
        #self.bert_vocab_dict = {}
        #self.cased=cased
        #self.load_bert_vocab(vocab_path,self.bert_vocab_dict)
        self.model_type='bert'
        #self.model_type='roberta'
        if self.model_type in {"gpt2", "roberta"}:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True,do_lower_case=lowercase)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True,do_lower_case=lowercase)
        
        self.tokenizer.add_tokens(["arg1s","arg1e","gene1s","gene1e","chemical1s","chemical1e"])

        #load label
        self.label_2_index={}
        self.index_2_label={}
        self.label_table_size=0
        self.load_label_vocab(label_file,self.label_2_index,self.index_2_label)
        self.label_table_size=len(self.label_2_index)
        self.vocab_len=len(self.tokenizer)
       
    def load_label_vocab(self,fea_file,fea_index,index_2_label):
        
        fin=open(fea_file,'r',encoding='utf-8')
        all_text=fin.read().strip().split('\n')
        fin.close()
        for i in range(0,len(all_text)):
            fea_index[all_text[i]]=i
            index_2_label[str(i)]=all_text[i]
            

            
    def generate_label_list(self,bert_tokens,labels,word_index):
        label_list=['O']*len(word_index)
        label_i=0
        if len(word_index)!=len(bert_tokens):
            print('index != tokens',word_index,bert_tokens)
            sys.exit()
        last_word_index=0
        for i in range(0,len(word_index)):
            if word_index[i]==None:
                pass
            else:
                label_list[i]=labels[word_index[i]]
        
        label_list_index=[]
        bert_text_label=[]
        for i in range(0,len(bert_tokens)):
            label_list_index.append(self.label_2_index[label_list[i]])
            bert_text_label.append([bert_tokens[i],label_list[i]])
        # for label in labels:
        #     temp_label=[0]*self.label_table_size
        #     temp_label[self.label_2_index[label]]=1
        #     label_list.append(temp_label)
        #print(bert_text_label)
        return label_list_index,bert_text_label
    
    
    def load_data_hugface(self,instances, labels,  word_max_len=100, label_type='crf',training_set=False):
    
        x_index=[]
        x_seg=[]
        x_mask=[]
        y_list=[]
        bert_text_labels=[]
        max_len=0
        over_num=0
        maxT=word_max_len
        ave_len=0

        #print('instances:', instances)
        #print('labels:',labels)
        
        
        for sentence in instances:                           
            sentence_text_list=[]
            label_list=[]
            for j in range(0,len(sentence)):
                sentence_text_list.append(sentence[j][0])
                label_list.append(sentence[j][-1])

            token_result=self.tokenizer(
                sentence_text_list,
                max_length=word_max_len,
                truncation=True,is_split_into_words=True)
            
            bert_tokens=self.tokenizer.convert_ids_to_tokens(token_result['input_ids'])
            word_index=token_result.word_ids(batch_index=0)
            ave_len+=len(bert_tokens)
            if len(sentence_text_list)>max_len:
                max_len=len(sentence_text_list)
            if len(bert_tokens)==maxT:
                over_num+=1

            x_index.append(token_result['input_ids'])
            if self.model_type in {"gpt2", "roberta"}:
                x_seg.append([0]*len(token_result['input_ids']))
            else:
                x_seg.append(token_result['token_type_ids'])
            x_mask.append(token_result['attention_mask'])

            #print('label:',label_list)
            label_list,bert_text_label=self.generate_label_list(bert_tokens,label_list,word_index)
            #print('\nlabel list:',label_list)
            #print('\nbert_text_label:',bert_text_label)
            #sys.exit()
            y_list.append(label_list)
            #print(y_list)
            bert_text_labels.append(bert_text_label)

        
        x1_np = pad_sequences(x_index, word_max_len, value=0, padding='post',truncating='post')  # right padding
        x2_np = pad_sequences(x_seg, word_max_len, value=0, padding='post',truncating='post')
        x3_np = pad_sequences(x_mask, word_max_len, value=0, padding='post',truncating='post')
        y_np = pad_sequences(y_list, word_max_len, value=0, padding='post',truncating='post')
        #print('x1_np:',x1_np)
        #print('\nx2_np:',x2_np)
        #print('\ny_np:',y_np)
        #print('\nbert_text:',bert_text_labels)
        #print('bert max len:',max_len,',Over',maxT,':',over_num,'ave len:',ave_len/len(instances),'total:',len(instances))
        if label_type=='onehot': 
            y_np = np.eye(len(labels), dtype='float32')[y_np]
        elif label_type=='softmax':
            y_np = np.expand_dims(y_np, 2)
        elif label_type=='crf':
            pass
        
        if training_set:
            #class_weight = {0: 1.0, 1: 3.597, 2: 4.106, 3: 4.004, 4: 7.312, 5: 3.956, 6: 8.046, 7: 2.254, 8: 3.612, 9: 3.715, 10: 8.170, 11: 3.239, 12: 3.066, 13: 3.654}
            class_weight = {0: 1.0, 1: 3.597, 2: 4.106, 3: 4.004, 4: 7.312, 5: 8.956, 6: 8.046, 7: 8.254, 8: 3.612, 9: 3.715, 10: 8.170, 11: 8.239, 12: 8.066, 13: 3.654}
            print('\n.......sample weight:',class_weight)
            sample_weight=[]
            for _line in y_list:
                _tempw=[]
                for _y in _line:
                    _tempw.append(class_weight[_y])
                sample_weight.append(_tempw)
            samplew_np = pad_sequences(sample_weight, word_max_len, value=0,dtype="float32", padding='post',truncating='post')
            return [x1_np, x2_np,x3_np], y_np,bert_text_labels, samplew_np
        
        return [x1_np, x2_np,x3_np], y_np,bert_text_labels  


            

if __name__ == '__main__':
    pass
    
 
            

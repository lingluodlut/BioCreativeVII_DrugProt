# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 14:59:07 2021

@author: luol2
"""

import argparse
import os
import time
import re
import io
import bioc
import subprocess

from tensorflow.keras.models import load_model
from model_sl import HUGFACE_SL
from processing_data import ml_intext_fn,out_BIO_fn,out_BIO_BERT_fn,out_BIO_BERT_softmax_fn,out_BIO_BERT_softmax_score_fn
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import sys

import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize',package='craft') #package='craft'


REL_ENT={'arg1':'GENE',
         'arg2':'CHEMICAL'}

entity_tag={'arg1':['arg1s','arg1e'],
            'gene':['gene1s','gene1e'],
            'chemical':['chemical1s','chemical1e']
    }

def ssplit_token(doc_text):
    
    fout_text=''
    lines=doc_text.split('\n')
    ori_text=lines[0]
    pmid=lines[1].split('\t')[0]
    entity_all=[]   #[[seg0,seg1,...,],[]]
    for i in range(1,len(lines)):
        seg=lines[i].split('\t')
        if len(seg)==6:
            entity_all.append(seg)
           
    #ssplit token
    doc_stanza = nlp(ori_text)
    token_text=''
    for sent in doc_stanza.sentences:
        for word in sent.words:
            if word.text==' ':
                print('token is blank!')
            token_text+=word.text+' '
        token_text=token_text+'    '  #sentence split by four blank
        
    #ori_index map token_index
    index_map=[-1]*len(ori_text)
    j=0
    space_list=[' ',chr(160),chr(8201),chr(8194),chr(8197),chr(8202)] #空格有好几种，
    for i in range(0,len(ori_text)):
        if ori_text[i] in space_list:
            pass
        elif ori_text[i]==token_text[j]:
            index_map[i]=j
            j+=1
        else:
            j+=1
            temp_log=j
            try:
                while(ori_text[i]!=token_text[j]):
                    j+=1
            except:
                print('doc',doc_text)
                print('token_text:',token_text)
                print('error:',ori_text[i-10:i+10],'i:',ori_text[i],'j:',token_text[temp_log],',',token_text[temp_log-10:temp_log+10])
                print(ord(ori_text[i]),ord(' '))
                sys.exit()
            index_map[i]=j
            j+=1
       
    fout_text=token_text+'\n'
    for ele in entity_all:
        new_ents=index_map[int(ele[3])]
        new_ente=index_map[int(ele[4])-1]+1
        new_ent=token_text[new_ents:new_ente]
        old_ent=ele[-1]
        fout_text+=ele[0]+'\t'+ele[1]+'\t'+ele[2]+'\t'+str(new_ents)+'\t'+str(new_ente)+'\t'+new_ent+'\n'

    fout_text+='\n'
    return fout_text

def input_preprocess_notoken(doc_text):
    final_input=[]
    final_id=[]
    
    lines=doc_text.strip().split('\n')
    token_text=lines[0]
    pmid=lines[1].split('\t')[0]
    entity_arg1=[]
    entity_all=[]
    for i in range(1,len(lines)):
        seg=lines[i].split('\t')
        if len(seg)==6:
            if seg[2]==REL_ENT['arg1']:
                entity_arg1.append(seg[1])
            entity_all.append(seg)
    
    
    #print(token_text)
    #print(entity_chemical)
    #generate input instance
    for cur_ele in entity_arg1:
        
        #1. drop nest entity
        nest_list=[entity_all[0]]
        max_eid=int(entity_all[0][4])
        entity_nonest=[]  #非嵌套实体列表
        for i in range(1,len(entity_all)):
            if int(entity_all[i][3])>=max_eid:
                if len(nest_list)==1:
                    entity_nonest.append(nest_list[0])
                    nest_list=[]
                    nest_list.append(entity_all[i])
                    max_eid=int(entity_all[i][4])
                else:
                    # print('nest:',nest_list)
                    tem=get_one_entity(nest_list,cur_ele)#find max entity，由于存在实体同时是药物和蛋白，所以根据关系去掉一个
                    # print('max:',tem)
                    entity_nonest.append(tem)
                    nest_list=[]
                    nest_list.append(entity_all[i])
                    max_eid=int(entity_all[i][4])                    
            else:
                nest_list.append(entity_all[i])
                if int(entity_all[i][4])>max_eid:
                    max_eid=int(entity_all[i][4])
        if nest_list!=[]:
            if len(nest_list)==1:
                entity_nonest.append(nest_list[0])
            else:
                # print('nest:',nest_list)
                tem=get_one_entity(nest_list,cur_ele)#find max entity
                # print('max:',tem)
                entity_nonest.append(tem)
        #2. ner label text
        ner_text=''
        text_sid=0
        #print('nonest:',entity_nonest)
        for ele_nonest in entity_nonest:
            ent_id=ele_nonest[1]
            ent_sid=int(ele_nonest[3])
            ent_eid=int(ele_nonest[4])
            # print('sid,eid:',ent_sid,ent_eid)
            ent_text=ele_nonest[5]
            if ent_text.find('     ')>=0:#revise the ssplit error at tokenize
                # print('ssplit error:',ent_text)
                ent_text=ent_text.replace('     ','')
            ent_type=ele_nonest[2]
            if ent_sid>=text_sid:
                # if token_text[ent_sid:ent_eid]!=ent_text:
                #     print('error!index_text,entext:',token_text[ent_sid:ent_eid],ent_text)
                if cur_ele==ent_id:    
                    ner_text+=token_text[text_sid:ent_sid]+' '+ent_id+'|'+entity_tag['arg1'][0]+' '+ent_text+' '+entity_tag['arg1'][1]+' '
                else:
                    ner_text+=token_text[text_sid:ent_sid]+' '+ent_id+'|'+entity_tag[ent_type.lower()][0]+' '+ent_text+' '+entity_tag[ent_type.lower()][1]+' '
                text_sid=ent_eid                                      
        ner_text+=token_text[text_sid:]
        ner_text=ner_text.replace('     ','<EOS>')
        ner_text=' '.join(ner_text.split())
        #print('\nner_text:',ner_text)
        
        #3. produce input
        sentences=ner_text.split('<EOS>')
        ins_i=0
        while (ins_i<len(sentences)):
            # print(i)
            tag,sen_tokens,ent_num=check_entity_pos(sentences[ins_i].strip())
            new_text=sentences[ins_i].strip()
            # print(new_text)
            while tag==-1:
                ins_i+=1
                new_text=new_text+' '+sentences[ins_i].strip()          
                tag,sen_tokens,ent_num=check_entity_pos(new_text) 
                print(tag,sen_tokens)
            # print('ent_label:',ent_label,tag)
            # print('seg:',seg)
            #print(tag,sen_tokens,ent_num)
            if ent_num['arg1']>0 and ent_num[REL_ENT['arg2'].lower()]>0:    
                temp_input=[]
                temp_id=[]
                for sen_token in sen_tokens:
                    if sen_token.find(entity_tag['arg1'][0])>=0:
                        en_id=sen_token.split('|')[0]
                        temp_id.append(en_id)
                        temp_input.append(entity_tag['arg1'][0]+'\tO')
                    elif sen_token.find(entity_tag['gene'][0])>=0:
                        en_id=sen_token.split('|')[0]
                        temp_id.append(en_id)
                        temp_input.append(entity_tag['gene'][0]+'\tO')
                    elif sen_token.find(entity_tag['chemical'][0])>=0:
                        en_id=sen_token.split('|')[0]
                        temp_id.append(en_id)
                        temp_input.append(entity_tag['chemical'][0]+'\tO')
                    else:
                        if sen_token=='':
                            print('token is none!error!')
                        else:
                            temp_input.append(sen_token+'\tO')
                final_input.append('\n'.join(temp_input))
                final_id.append(temp_id)
                break
            ins_i+=1
        # print(entity_nonest)
    return final_input,final_id,pmid


def check_entity_pos(line):
    
    seg=line.split(' ')
    stack_ent=[] 
    # print(seg)
    entity_num={'arg1':0,'gene':0,'chemical':0}
    for i in range(0,len(seg)):
        if seg[i].find(entity_tag['gene'][0])>=0:
            entity_num['gene']+=1
            stack_ent.append(seg[i])
        elif seg[i].find(entity_tag['chemical'][0])>=0:
            entity_num['chemical']+=1
            stack_ent.append(seg[i])
            # print(stack_ent)
        elif seg[i].find(entity_tag['arg1'][0])>=0:
            entity_num['arg1']+=1
            stack_ent.append(seg[i])        
        elif seg[i].find(entity_tag['arg1'][1])>=0  or seg[i].find(entity_tag['gene'][1])>=0 or seg[i].find(entity_tag['chemical'][1])>=0:
            stack_ent.pop()
    if stack_ent!=[]:
        print('entity no match!',stack_ent)
        return(-1,seg,entity_num)
    
    else:
        return(1,seg,entity_num) 

def get_one_entity(nest_list,cur_ele):
    max_len=0
    max_entity=[]
    for i in range(0, len(nest_list)):
        if nest_list[i][1]==cur_ele:#current relation entity
            max_entity=nest_list[i]
            break
        length=int(nest_list[i][4])-int(nest_list[i][3])
        if max_entity==[]: #first entity
            max_len=length
            max_entity=nest_list[i]
        else:
            if length>max_len:
                if max_entity[2]==REL_ENT['arg1']:
                    max_len=length
                    max_entity=nest_list[i]
                else:
                    if nest_list[i][2]==REL_ENT['arg2']:
                        max_len=length
                        max_entity=nest_list[i] 
            else:
                if max_entity[2]==REL_ENT['arg1'] and nest_list[i][2]==REL_ENT['arg2']:
                    max_len=length
                    max_entity=nest_list[i]
                
    return max_entity

def ml_tagging(ml_input,nn_model):

    if nn_model.model_type=='NN':
        #print('BiLSTM-CRF tagging')
        test,test_label = ml_intext_fn(ml_input)
        test_x, test_y = nn_model.rep.represent_instances_fea(test,test_label,word_max_len=nn_model.hyper['sen_max'],char_max_len=nn_model.hyper['word_max'],cased=False,label_type='softmax')
        input_test = []
    
        if nn_model.fea_dict['word'] == 1:
            input_test.append(test_x[0])
    
        if nn_model.fea_dict['char'] == 1:
            input_test.append(test_x[1])
    
    
        if nn_model.fea_dict['pos'] == 1:
            input_test.append(test_x[2])
    
        test_pre = nn_model.model.predict(input_test,batch_size=256)
        ml_out=out_BIO_fn(test_pre,test,nn_model.rep.index_2_label)

    
    elif nn_model.model_type=='HUGFACE':

        test_set,test_label = ml_intext_fn(ml_input)
        test_x,test_y, test_bert_text_label=nn_model.rep.load_data_hugface(test_set,test_label,word_max_len=nn_model.maxlen,label_type='softmax')
        test_pre = nn_model.model.predict(test_x)
        ml_out=out_BIO_BERT_softmax_score_fn(test_pre,test_bert_text_label,nn_model.rep.index_2_label)

    return ml_out

def output_rel(ml_output,entity_map,pmid):
    fin=io.StringIO(ml_output)
    # fin=open('//panfs/pan1/bionlplab/luol2/BC7DrugProt/results/drugprot_train_d1_pre.conll','r',encoding='utf-8')
    # pmid='1'
    # entity_map=[['T1','T6','T7'],['T2','T5','T8']]
    alltexts=fin.read().strip().split('\n\n')
    fin.close()
    final_out=[]
    for sen_id,sentence in enumerate(alltexts):
        tokens=sentence.split('\n')
        entity_id=0
        token_id=0
        arg1=''
        arg2_list=[] #[[ID, type,score],[id,type,score]]
        while (token_id<len(tokens)):
            seg=tokens[token_id].split('\t')
            if seg[0]==entity_tag['arg1'][0]:
                arg1=entity_map[sen_id][entity_id]
                entity_id+=1
                token_id+=1
                if token_id >=len(tokens):
                    break
                seg=tokens[token_id].split('\t')
                while seg[0]!=entity_tag['arg1'][1]:
                    token_id+=1
                    if token_id >=len(tokens):
                        break
                    seg=tokens[token_id].split('\t')
            elif seg[0]==entity_tag[REL_ENT['arg2'].lower()][0]:
                temp_rel=seg[2]
                temp_score=seg[-1]
                arg2_id=entity_map[sen_id][entity_id]
                entity_id+=1
                token_id+=1
                if token_id >=len(tokens):
                    break
                seg=tokens[token_id].split('\t')
                while seg[0]!=entity_tag[REL_ENT['arg2'].lower()][1]:
                    token_id+=1
                    if token_id >=len(tokens):
                        break
                    seg=tokens[token_id].split('\t')
                    if seg[2].find('ARG2-')>=0 and temp_rel.find('ARG2-')<0:
                        temp_rel=seg[2]
                        temp_score=seg[-1]
                if temp_rel.find('ARG2-')>=0:
                    arg2_list.append([arg2_id,temp_rel[5:],temp_score])
            elif seg[0]==entity_tag[REL_ENT['arg1'].lower()][0]:
                entity_id+=1
                token_id+=1
                if token_id >=len(tokens):
                    break
                seg=tokens[token_id].split('\t')
                while seg[0]!=entity_tag[REL_ENT['arg1'].lower()][1]:
                    token_id+=1
                    if token_id >=len(tokens):
                        break
                    seg=tokens[token_id].split('\t')

            else:
                pass
            token_id+=1
        #print(arg1,arg2_list)
        if arg2_list!=[] and arg1!='':
            for arg2_ele in arg2_list:
                if REL_ENT['arg1']=='CHEMICAL':
                    final_out.append([pmid,arg2_ele[1],'Arg1:'+arg1,'Arg2:'+arg2_ele[0],arg2_ele[-1]])
                else:
                    final_out.append([pmid,arg2_ele[1],'Arg1:'+arg2_ele[0],'Arg2:'+arg1,arg2_ele[-1]])
    return(final_out)

    
def NER_Tag(doc_in,nn_model,outfile):
    
    #1. preprocess input, input_text:conll格式, input_entity：相应的实体列表, split, token
    #print(doc_in)
    doc_in_token=ssplit_token(doc_in)
    input_text,input_entity,pmid=input_preprocess_notoken(doc_in_token)
    print('pmid:',pmid)

   
    #2. ml tagging
    if input_text!=[]:
        ml_pre=ml_tagging(input_text,nn_model)
        #print('\noutput:')
        #print(ml_pre)
        #3.generate output
        final_output=output_rel(ml_pre,input_entity,pmid)
    else:
        final_output=[]
    return final_output
    

def main(infile,outfile,modelfile):
    
    print('loading models........')    
        
    
    vocabfiles={'labelfile':'../vocab/IO_label.vocab',
                'checkpoint_path':'../model/BioM-ELECTRA-Large-Discriminator/',
                'lowercase':True}
       
    nn_model=HUGFACE_SL(vocabfiles)
    nn_model.build_encoder()
    nn_model.build_softmax_decoder()
    nn_model.load_model(modelfile)
   
    #tagging text
    print("begin tagging........")
    start_time=time.time()
    
    fin=open(infile,'r',encoding='utf-8')
    all_in=fin.read().strip().split('\n\n')
    fin.close()
    with open(outfile, 'w',encoding='utf-8') as fout:
        i=0
        for doc in all_in:
            i+=1
            print(i)
            pre_result=NER_Tag(doc, nn_model,outfile)
            for ele in pre_result:
                fout.write('\t'.join(ele[:-1])+'\n')
    
    print('tag done:',time.time()-start_time)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='drug-protein extraction, python DrugProt_Tagging_PD.py -i infile -o outfile')
    parser.add_argument('--infile', '-i', help="input file",default='../example/example_input.tsv')
    parser.add_argument('--outfile', '-o', help="output file",default='../example/example_out.tsv')
    args = parser.parse_args()

    modelfile='../model/BioM-ELECTRAL-softmax-devES_GC_final.h5'

    main(args.infile,args.outfile,modelfile)
    
    

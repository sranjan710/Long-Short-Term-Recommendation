import pandas as pd
import os
import datetime
import numpy as np
import pickle

from tools import log_time_delta
import time
from multiprocessing import Pool
from multiprocessing import freeze_support
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
import math
from config import Singleton
import sklearn
import itertools
import tensorflow as tf
import random
from tqdm import tqdm
mp=False

class DataHelper():
    def __init__(self,conf,mode="run"):
        self.conf=conf

        #dataset loaded
        self.data = self.loadData()


        '''train and test data split---->'''
        self.train= self.data[self.data.days<0]
        self.test= self.data[self.data.days>=0]

        np.savetxt(r'tmp/train.txt', self.train.values, fmt='%d')
        np.savetxt(r'tmp/test.txt', self.test.values, fmt='%d')



        '''user and item count'''
        self.u_cnt= self.data ["uid"].max()+1
        self.i_cnt= self.data ["itemid"].max()+1
        print("Number of users-->{}".format(self.u_cnt))
        print("Number of items-->{}".format(self.i_cnt))



        ''' in user_dict,  with the same userid and user_granularity it stores itemid and rating
            in item_dict,  with the same userid and user_granularity it stores uid and rating '''
        self.user_dict,self.item_dict=self.getdicts()

        #np.savetxt(r'tmp/user_dict.txt', self.user_dict.values, fmt='%d')
        #np.savetxt(r'tmp/item_dict.txt', self.item_dict.values, fmt='%d')


        #finding unique users
        self.users=set(self.data["uid"].unique())


        #self.train_users=set(self.train["uid"].unique())
        #self.items = set([i for i in range(self.i_cnt)])

        get_pos_items=lambda group: set(group[group.rating>(4.99 if  self.conf.rating_flag else 0.5)]["itemid"].tolist())

        self.pos_items=self.train.groupby("uid").apply(get_pos_items)

        user_item_pos_rating_time_dict= lambda group:{item:time for i,(item,time)  in group[group.rating>(4.99 if  self.conf.rating_flag else 0.5)][["itemid","user_granularity"]].iterrows()}

        self.user_item_pos_rating_time_dict=self.train.groupby("uid").apply(user_item_pos_rating_time_dict).to_dict()
        np.savetxt(r'tmp/user_item_pos_rating_time_dict.txt', self.item_dict.values, fmt='%d')

        self.test_pos_items=self.test.groupby("uid").apply(get_pos_items).to_dict()


    def create_dirs(self,dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # @log_time_delta
    def loadData(self):
        self.create_dirs("tmp")
        dataset_pkl = "tmp/"+self.conf.dataset +"_"+self.conf.split_data+("" if  self.conf.rating_flag else "_binary")+".pkl"
        if os.path.exists(dataset_pkl):
            print("data load over")
            return pickle.load(open(dataset_pkl, 'rb'))

        print("build data...")
        data_dir="data/%s"% self.conf.dataset
        filename = os.path.join(data_dir, self.conf.train_file_name)
        print(filename)
        df = pd.read_csv(filename,sep="\t", names=["uid","itemid","rating","timestamp"])

        df = df.sort_values(["uid","itemid"])

        print("there are %d users in this dataset" %(df ["uid"].max()+1))

        y,m,d = (int(i) for i in self.conf.split_data.split("-"))

        df["days"] = (pd.to_datetime(df["timestamp"], unit='s') - pd.datetime(y,m,d )).dt.days

        df["item_granularity"] = df["days"] // self.conf.item_delta   # //means floor div
        df["user_granularity"] = df["days"] // self.conf.user_delta   # //means floor div

        if self.conf.threshold > 0: # remove the users while the rating of them is lower than threshold
            counts_df = pd.DataFrame(df.groupby('uid').size().rename('counts'))

            users = set(counts_df[counts_df.counts>self.conf.threshold].index)

            df = df[df.uid.isin(users)]

        if not self.conf.rating_flag :
            df["rating"]=(df["rating"]>3.99).astype('int')
            df=df[df.rating >  0.5]
            #np.savetxt(r'tmp/before.txt', df.values, fmt='%d')

            #np.savetxt(r'tmp/after.txt', df.values, fmt='%d')

#       re-arrange the user and item index from zero
        df['u_original'] = df['uid'].astype('category')
        df['i_original'] = df['itemid'].astype('category')

        df['uid'] = df['u_original'].cat.codes
        df['itemid'] = df['i_original'].cat.codes

        df = df.drop('u_original', 1)
        df = df.drop('i_original', 1)


        pickle.dump(df, open(dataset_pkl, 'wb'),protocol=2)
        return df

    def user_windows_apply(self,group,user_dict):
        uid=(int(group["uid"].mode()))

        # user_dict[uid]= len(group["days"].unique())
        user_dict.setdefault(uid,{})
        for user_granularity in group["user_granularity"]:
            # print (group[group.user_granularity==user_granularity])
            if  self.conf.rating_flag:
                user_dict[uid][user_granularity]= group[group.user_granularity==user_granularity][["itemid","rating"]]
            else:
                user_dict[uid][user_granularity]= group[(group.user_granularity==user_granularity) & (group.rating>0)][["itemid","rating"]]
        return len(group["user_granularity"].unique())

    def item_windows_apply(self,group,item_dict):
        itemid=(int(group["itemid"].mode()))
        # user_dict[uid]= len(group["days"].unique())
        item_dict.setdefault(itemid,{})
        for item_granularity in group["item_granularity"]:
            # print (group[group.user_granularity==user_granularity])
            if  self.conf.rating_flag:
                item_dict[itemid][item_granularity]= group[group.item_granularity==item_granularity][["uid","rating"]]
            else:
                item_dict[itemid][item_granularity]= group[(group.item_granularity==item_granularity)  & (group.rating>0)][["uid","rating"]]
            # print (item_dict[itemid][item_granularity])
        return len(group["item_granularity"].unique())

    # @log_time_delta
    def getdicts(self):
        dict_pkl = "tmp/user_item_"+self.conf.dataset+("" if  self.conf.rating_flag else "_binary")+".pkl"
        if os.path.exists(dict_pkl):
            start=time.time()
            import gc
            gc.disable()
            user_dict,item_dict= pickle.load(open(dict_pkl, 'rb'))
            gc.enable()
            print( "load dict cost  time: %.5f "%( time.time() - start))
        else:
            print("build data...")
            user_dict,item_dict={},{}
            user_windows = self.data.groupby("uid").apply(self.user_windows_apply,user_dict=user_dict)
            item_windows = self.data.groupby("itemid").apply(self.item_windows_apply,item_dict=item_dict)
            pickle.dump([user_dict,item_dict], open(dict_pkl, 'wb'),protocol=2)

        return user_dict,item_dict

    def getSeqInTime(self,userid,itemid,chosen_t=0):
        #print(userid,itemid,chosen_t,choice_type)

        u_seqs,i_seqs=[],[]
        #print("chosen_t--->",chosen_t)
        #print(userid,itemid)
        for i in range(chosen_t-self.conf.user_windows_size,chosen_t):
            u_seqs.append(self.user_dict[userid].get(i,None))
            i_seqs.append(self.item_dict[itemid].get(i,None))
        if self.conf.is_sparse:
            return self.getUserVector(u_seqs),self.getItemVector(i_seqs)
        else:

            return self.getUserVector_raw(u_seqs),self.getItemVector_raw(i_seqs)


    def getUserVector_raw(self,user_sets):
        u_seqs=[]
        for user_set in user_sets:
            u_seq=[0]*(self.i_cnt)

            if not user_set is None:
                for index,row in user_set.iterrows():
                    u_seq[row["itemid"]]=row["rating"]
            u_seqs.append(u_seq)
        return np.array(u_seqs)


    def getItemVector_raw(self,item_sets):
        i_seqs=[]
        for item_set in item_sets:
            i_seq=[0]*(self.u_cnt)
            if not item_set is None:
                for index,row in item_set.iterrows():
                   i_seq[row["uid"]]=row["rating"]
            i_seqs.append(i_seq)
        return np.array(i_seqs)


    def getItemVector(self,item_sets):
        rows=[]
        cols=[]
        datas=[]
        for index_i,item_set in enumerate(item_sets):
            if not item_set is None:
                for index_j,row in item_set.iterrows():
                    rows.append(index_i)
                    cols.append(row["uid"])
                    datas.append(row["rating"])
        if self.conf.sparse_tensor:
            return ( rows,cols ,datas)
        result=csr_matrix((datas, (rows, cols)), shape=(self.conf.user_windows_size, self.u_cnt))
        return result

    def getUserVector(self,user_sets):
        rows=[]
        cols=[]
        datas=[]
        for index_i,user_set in enumerate(user_sets):
            if not user_set is None:
                for index,row in user_set.iterrows():
                    rows.append(index_i)
                    cols.append(row["itemid"])
                    datas.append(row["rating"])
        if self.conf.sparse_tensor:
            return ( rows,cols ,datas)
        return csr_matrix((datas, (rows, cols)), shape=(self.conf.user_windows_size, self.i_cnt))


    def get_user_sparse_input(self,user_sequence):
        _indices,_values=[],[]
        for index,(cols,rows,values)  in enumerate(user_sequence):
            _indices.extend([index,x,y]  for x,y in zip(cols,rows) )   #sorted(zip(cols,rows),key =lambda x:x[0]*2000+x[1] )
            _values.extend(values)
        if len(_indices)==0:
            return ([[0,0,0]],[0],[len(user_sequence),self.conf.user_windows_size,self.i_cnt ])
        user_input= (_indices,_values,[len(user_sequence),self.conf.user_windows_size,self.i_cnt ])
        return user_input
    def get_item_sparse_input(self,item_sequence):
        _indices,_values=[],[]
        for index,(cols,rows,values)  in enumerate(item_sequence):
            _indices.extend([index,x,y]  for x,y in  zip(cols,rows))
            _values.extend(values)
        if len(_indices)==0:
            return ([[0,0,0]],[0],[len(item_sequence),self.conf.user_windows_size,self.u_cnt ])
        item_input= (_indices,_values,[len(item_sequence),self.conf.user_windows_size,self.u_cnt ])
        return item_input
    def get_sparse_intput(self,user_sequence,item_sequence):
        user_input=self.get_user_sparse_input(user_sequence)
        item_input=self.get_item_sparse_input(item_sequence)

        return user_input,item_input

    def sparse2dense(sparse):
        return sparse.toarray()

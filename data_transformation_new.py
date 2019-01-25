# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:46:47 2019

@author: anukapoor
"""

import pandas as pd 

from flask import Flask,render_template,url_for,request

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
import numpy as np
import os
import  re
import string
from sklearn.feature_extraction import stop_words
from sklearn.externals import joblib


def transform_data(train_df):

#    test=pd.read_csv("data_desc_name.csv", sep=',', encoding='latin-1',header=0)
    
    def split_cat(row):
        try: 
            cat_1,cat_2,cat_3=row.split('/')
            return cat_1,cat_2,cat_3
        except:
            return np.nan,np.nan,np.nan
    # Here we are using zip(* something ) to unzip list of tuples
    train_df["cat_1"], train_df["cat_2"], train_df["cat_3"] = zip(*train_df.category_name.apply(lambda val: split_cat(val)))
    data=train_df
    
        
    with open('cat1_dict.p', 'rb') as fp:
        cat1_dict = pickle.load(fp)
    with open('cat2_dict.p', 'rb') as fp:
        cat2_dict = pickle.load(fp)
    with open('cat3_dict.p', 'rb') as fp:
        cat3_dict = pickle.load(fp)    
    
    def cat_lab(row,cat1_dict = cat1_dict, cat2_dict = cat2_dict, cat3_dict = cat3_dict):
        """function to give cat label for cat1/2/3"""
        txt1 = row['cat_1']
        txt2 = row['cat_2']
        txt3 = row['cat_3']
        return cat1_dict[txt1], cat2_dict[txt2], cat3_dict[txt3]
    
    data["cat_1_label"], data["cat_2_label"], data["cat_3_lable"] = zip(*data.apply(lambda val: cat_lab(val), axis =1))
    
    data['brand_name_avail']=np.where(data['brand_name'].isnull(),0,1)
    data['cat_name_avail']=np.where(data['category_name'].isnull(),0,1)
    
    
    with open('brand_dict.p', 'rb') as fp:
        brand_dict = pickle.load(fp)    
        
    def brand_label(row):
        """function to assign brand label"""
        try:
            return brand_dict[row]
        except:
            return np.nan
    
    data['brand_label'] = data.brand_name.apply(lambda row: brand_label(row))
    data['is_description'] = np.where(data.item_description=="No description yet",0,1)
    
    brand_name=pd.read_csv('brand_name.csv')
    data=pd.merge(data,brand_name[['brand_name','brand_group']],left_on=['brand_name'],right_on=['brand_name'],how='left')
    
    
    def wordCount(text):
        # convert to lower case and strip regex
        try:
             # convert to lower case and strip regex
            text = text.lower()
            regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
            txt = regex.sub(" ", text)
            # tokenize
            # words = nltk.word_tokenize(clean_txt)
            # remove words in stop words
            words = [w for w in txt.split(" ") \
                     if not w in stop_words.ENGLISH_STOP_WORDS and len(w)>3]
            return len(words)
        except: 
            return 0
        
    data['desc_len'] = data['item_description'].apply(lambda x: wordCount(x))
    
    
      
    
    ### item desc
    from sklearn.externals import joblib
    tfidf_vec = joblib.load('tfidf_vec_desc_test.pkl') 

#    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
#    
#    full_tfidf = tfidf_vec.fit_transform(test['item_description'].values.tolist() )
    
    train_tfidf = tfidf_vec.transform(data['item_description'].values.tolist())    
    

    svd_obj = joblib.load('svd_obj_desc_test.pkl')       
    n_comp = 40
#    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
#    svd_obj.fit(full_tfidf)
    #    with open('svd_obj_desc.p', 'rb') as fp:
    #        svd_obj  = pickle.load(fp) 
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
        
    train_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
    data = pd.concat([data, train_svd], axis=1)
        
    # Products
    
    from sklearn.externals import joblib
    tfidf_vec_prod = joblib.load('tfidf_vec_name_test.pkl') 
#    tfidf_vec_prod = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
#    full_tfidf_prod = tfidf_vec_prod.fit_transform(test['name'].values.tolist() )
    train_tfidf_prod = tfidf_vec_prod.transform(data['name'].values.tolist())
    
    
#    n_comp = 40
#    svd_obj_prod = TruncatedSVD(n_components=n_comp, algorithm='arpack')
#    svd_obj_prod.fit(full_tfidf_prod)
    #    with open('svd_obj_name.p', 'rb') as fp:
    #        svd_obj  = pickle.load(fp) 
    svd_obj_prod = joblib.load('svd_obj_name_test.pkl') 
    train_svd_prod = pd.DataFrame(svd_obj_prod.transform(train_tfidf_prod))
    
        
    train_svd_prod.columns = ['svd_name_'+str(i) for i in range(n_comp)]
    
    data = pd.concat([data, train_svd_prod], axis=1)
    
    max_min_med=pd.read_csv("max_min_med.csv",sep=',', encoding='latin-1')
    
    
    data=pd.merge(data,
                     max_min_med,
                     on=['cat_1','cat_2','cat_3'])
    return data;
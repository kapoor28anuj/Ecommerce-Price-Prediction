import math
import xgboost as xgb
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
import numpy as np
import pandas as pd
import os


#os.chdir('C:\\Users\\anukapoor\\Desktop\\Ecommerce_prediction_app')
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
import  re
import string
from sklearn.feature_extraction import stop_words
from sklearn.externals import joblib

from data_transformation_new import transform_data

feature_names=['item_condition_id', 'shipping', 'cat_1_label', 'cat_2_label', 'cat_3_lable', 
           'brand_name_avail', 'cat_name_avail', 'brand_label', 'brand_group', 'desc_len', 
           'svd_item_0', 'svd_item_1', 'svd_item_2', 'svd_item_3', 'svd_item_4', 'svd_item_5', 
           'svd_item_6', 'svd_item_7', 'svd_item_8', 'svd_item_9', 'svd_item_10', 'svd_item_11',
           'svd_item_12', 'svd_item_13', 'svd_item_14', 'svd_item_15', 'svd_item_16', 'svd_item_17',
           'svd_item_18', 'svd_item_19', 'svd_item_20', 'svd_item_21', 'svd_item_22', 'svd_item_23', 
           'svd_item_24', 'svd_item_25', 'svd_item_26', 'svd_item_27', 'svd_item_28', 'svd_item_29',
           'svd_item_30', 'svd_item_31', 'svd_item_32', 'svd_item_33', 'svd_item_34', 'svd_item_35', 
           'svd_item_36', 'svd_item_37', 'svd_item_38', 'svd_item_39', 'svd_name_0', 'svd_name_1', 
           'svd_name_2', 'svd_name_3', 'svd_name_4', 'svd_name_5', 'svd_name_6', 'svd_name_7', 
           'svd_name_8', 'svd_name_9', 'svd_name_10', 'svd_name_11', 'svd_name_12', 'svd_name_13', 
           'svd_name_14', 'svd_name_15', 'svd_name_16', 'svd_name_17', 'svd_name_18', 'svd_name_19',
           'svd_name_20', 'svd_name_21', 'svd_name_22', 'svd_name_23', 'svd_name_24', 'svd_name_25', 
           'svd_name_26', 'svd_name_27', 'svd_name_28', 'svd_name_29', 'svd_name_30', 'svd_name_31', 
           'svd_name_32', 'svd_name_33', 'svd_name_34', 'svd_name_35', 'svd_name_36', 'svd_name_37',
           'svd_name_38', 'svd_name_39', 'max_logprice', 'med_logprice']



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	xg_model = open('xgboost_model15min.pkl','rb')
	model_1 = joblib.load(xg_model)

	if request.method == 'POST':
         product_name = request.form['product_name']
         item_condition = request.form['item_condition']
         category_name = request.form['category_name']
         brand_name = request.form['brand_name']
         shipping = request.form['shipping']
         item_description = request.form['item_description']
         data = [[product_name,item_condition,category_name,brand_name,shipping,item_description]]
         df = pd.DataFrame(data,columns=[ 'name', 'item_condition_id', 'category_name', 'brand_name', 'shipping', 'item_description'])
         df_trans=transform_data(df)
         df_pred=df_trans[feature_names]
         dtest = xgb.DMatrix(df_pred.values)
         yvalid = model_1.predict(dtest)
         my_prediction = round(math.exp( yvalid ) ,2)
    
		
#		vect = cv.transform(data).toarray()
	
	return render_template('result.html',prediction = my_prediction)

#product_name
#item_condition
#category_name
#brand_name
#shipping
#item_description

if __name__ == '__main__':
	app.run(debug=True)
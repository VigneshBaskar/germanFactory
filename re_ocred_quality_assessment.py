
# coding: utf-8

# In[32]:

from __future__ import division
#%matplotlib inline
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pymysql
import scipy.stats
from sklearn import linear_model
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import os
import random
import itertools
import warnings
import shutil
warnings.filterwarnings("ignore")


# In[33]:


import logging
logger = logging.getLogger()
os.chdir(r'C:\Users\vbask\Documents\Darts_IP\Darts_IP\Factories\German_Factory_Code\germanFactory')
fhandler = logging.FileHandler(filename='mylog.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

logging.info('Logging starts for Re-OCRing')




# In[34]:

german_train_path="C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\English_Decisions_Distribution\\German Training files"

os.chdir(german_train_path)

X_array=pickle.load(open("X_array.p","rb"))
training_poor_fk=pickle.load(open("training_poor_fk.p","rb"))
training_poor_filename=pickle.load(open("training_poor_filename.p","rb"))
training_poor_text=pickle.load(open("training_poor_text.p","rb"))
training_poor_text_processed=pickle.load(open("training_poor_text_processed.p","rb"))
X_array_training_poor=pickle.load(open("X_array_training_poor.p","rb"))

training_wrong_lang_fk=pickle.load(open("training_wrong_lang_fk.p","rb"))
training_wrong_lang_filename=pickle.load(open("training_wrong_lang_filename.p","rb"))
training_wrong_lang_text=pickle.load(open("training_wrong_lang_text.p","rb"))
training_wrong_lang_text_processed=pickle.load(open("training_wrong_lang_text_processed.p","rb"))
X_array_training_wrong_lang=pickle.load(open("X_array_training_wrong_lang.p","rb"))

training_good_fk=pickle.load(open("training_good_fk.p","rb"))
training_good_filename=pickle.load(open("training_good_filename.p","rb"))
training_good_text=pickle.load(open("training_good_text.p","rb"))
training_good_text_processed=pickle.load(open("training_good_text_processed.p","rb"))
X_array_training_good=pickle.load(open("X_array_training_good.p","rb"))


# In[35]:

# plt.plot(X_array_training_good[:,1],X_array_training_good[:,0],'go',label="Good")
# plt.plot(X_array_training_wrong_lang[:,1],X_array_training_wrong_lang[:,0],'yo',label="Wrong language")
# plt.plot(X_array_training_poor[:,1],X_array_training_poor[:,0],'ro',label="Poor")

# plt.legend(loc='lower right')
# plt.xlabel('Length of tokens')
# plt.ylabel('% of tokens matching a dictionary')
# plt.title('Training files visualization')
# plt.xlim(0,4000)
# plt.show()


# In[36]:

X_array_training=np.vstack((X_array_training_poor,X_array_training_wrong_lang,X_array_training_good))
Y_array_training=[0 for i in range(len(X_array_training_poor))]+[0 for i in range(len(X_array_training_wrong_lang))]+[2 for i in range(len(X_array_training_good))]


# In[37]:

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

scaler=preprocessing.StandardScaler().fit(X_array_training)
X_array_training_scaled=scaler.transform(X_array_training)

#clf = svm.SVC(class_weight={0:7,2:1},kernel='rbf',gamma=3)
clf = svm.SVC(class_weight={0:7,2:1},kernel='linear')
clf.fit(X_array_training_scaled,Y_array_training)


# In[38]:

import io
from pdfminer.pdfinterp import PDFResourceManager,PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

def convert(fname, pages=None):
    if not pages:
        pareums=set()
    else:
        pareums=set(pages)
        
    output=io.BytesIO()
    manager=PDFResourceManager()
    converter=TextConverter(manager, output, codec='utf-8', laparams=LAParams())
    interpreter=PDFPageInterpreter(manager,converter)
    
    infile=file(fname,'rb')
    for page in PDFPage.get_pages(infile, pareums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text=output.getvalue()
    output.close
    return text


# In[39]:

import unicodedata
import sys

tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P'))
def remove_punctuation(text):
    return text.translate(tbl)

def extract_quality_features(text,dictionary_words,show=1):
    percent_tokens=[]
    len_tokens=[]
    text_processed=[]
    for i in np.arange(len(text)):
        temp=text[i].replace(u"\u2018", "").replace(u"\u2019", "").replace(u"\n"," ")
        tokens=temp.split()
        tokens=[remove_punctuation(word.lower()) for word in tokens]
        punctuations = re.compile(r'[-./?!,--&":;()|0-9]')
        tokens=[punctuations.sub("", word) for word in tokens if punctuations.sub("", word)]
        text_processed.append(" ".join(tokens))
        len_tokens.append(len(tokens))   

    vect=CountVectorizer(vocabulary=dictionary_words)
    X_counts=vect.transform(text_processed)
    X_counts_sum=X_counts.sum(axis=1)
    X_counts_sum = np.squeeze(np.asarray(X_counts_sum))

    percent_tokens=[]
    for i in range(len(text)):
        temp=X_counts_sum[i]/len_tokens[i]
        if np.isfinite(temp):
            percent_tokens.append(temp if temp<1 else 1)
        else:
            percent_tokens.append(0)
    if show:
        print "Do nothing"
    return percent_tokens,len_tokens,text_processed


# In[40]:

os.chdir(r'C:\Users\vbask\Documents\Darts_IP\Darts_IP\Dictionaries')
dictionary_words=pickle.load(open("latin_rus_dic.p","rb"))


# In[41]:

re_OCRed_text_path="C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\German_Factory\\4_b_Re_OCRed"
os.chdir(re_OCRed_text_path)

re_OCRed_text_filename=[f for f in os.listdir(re_OCRed_text_path) if os.path.isfile(f) and os.path.splitext(f)[1]==".pdf"]
print re_OCRed_text_filename
print len(re_OCRed_text_filename)


# In[42]:

import time
time1=time.time()
import os

os.chdir(re_OCRed_text_path)
re_OCRed_text=[]
erroneous_fnames=[]
non_erroneous_fnames=[]

for fname in re_OCRed_text_filename:
    try:
        print("Success "+fname)
        re_OCRed_text.append(convert(fname).decode('utf-8','ignore'))
        non_erroneous_fnames.append(fname)
        

    except:
        print('Some Error skipped with try')
        print fname
        erroneous_fnames.append(fname)

time2=time.time()
print('\n')
print (time2-time1)
print('Non OCR Prediciton through error')
print erroneous_fnames


# In[43]:

re_OCRed_text_filename=non_erroneous_fnames


# In[44]:

re_OCRed_text_percent_tokens,re_OCRed_text_len_tokens,re_OCRed_text_processed=extract_quality_features(re_OCRed_text,dictionary_words,0)


# In[45]:

X_array_re_ocred=np.vstack((np.asarray(re_OCRed_text_percent_tokens),np.asarray(re_OCRed_text_len_tokens))).T
X_array_re_ocred_scaled=scaler.transform(X_array_re_ocred)


# In[47]:

clf = svm.SVC(class_weight={0:7,2:1},kernel='rbf',probability=True)
clf.fit(X_array_training_scaled,Y_array_training)
X_array_re_ocred_pred_7=clf.predict(X_array_re_ocred_scaled)
X_array_re_ocred_pred=X_array_re_ocred_pred_7

# plt.plot(X_array_re_ocred[X_array_re_ocred_pred==2][:,1],X_array_re_ocred[X_array_re_ocred_pred==2][:,0],'g.')
# plt.plot(X_array_re_ocred[X_array_re_ocred_pred==0][:,1],X_array_re_ocred[X_array_re_ocred_pred==0][:,0],'r.')

# plt.xlabel('Test length of tokens')
# plt.ylabel('Test % of tokens matching a dictionary')
# plt.title('Decision boundary of Linear SVM for test data')

# plt.figure()
# plt.plot(X_array_re_ocred_scaled[X_array_re_ocred_pred==2][:,1],X_array_re_ocred_scaled[X_array_re_ocred_pred==2][:,0],'g.')
# plt.plot(X_array_re_ocred_scaled[X_array_re_ocred_pred==0][:,1],X_array_re_ocred_scaled[X_array_re_ocred_pred==0][:,0],'r.')

# plt.xlabel('Normalized test length of tokens')
# plt.ylabel('Normalized test % of tokens matching a dictionary')
# plt.title('Decision boundary of Linear SVM for normalized test data')



# In[48]:

X_array_test=pickle.load(open("C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\German_Factory\\1_Poor_downloaded\\X_array_test.p","rb"))
test_text_filename=pickle.load(open("C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\German_Factory\\1_Poor_downloaded\\test_text_filename.p","rb"))


# In[49]:

X_array_test_scaled=scaler.transform(X_array_test)


# In[50]:

improvements=[clf.predict_proba(X_array_re_ocred_scaled[re_OCRed_text_filename.index(re_OCRed_text_filename[index])])[0][1] -clf.predict_proba(X_array_test_scaled[test_text_filename.index(re_OCRed_text_filename[index])])[0][1] for index in range(len(re_OCRed_text_filename))]
X=re_OCRed_text_filename
Y=improvements
sorted_re_OCRed_text_filename=[x for (y,x) in sorted(zip(Y,X))]   


# In[51]:

print improvements
print re_OCRed_text_filename
print sorted_re_OCRed_text_filename
print sorted_re_OCRed_text_filename[-2:]


# In[52]:

num_imp=0
for index in range(len(re_OCRed_text_filename)):
    if clf.predict_proba(X_array_test_scaled[test_text_filename.index(re_OCRed_text_filename[index])])[0][1]<clf.predict_proba(X_array_re_ocred_scaled[re_OCRed_text_filename.index(re_OCRed_text_filename[index])])[0][1] :
        num_imp+=1


# In[53]:

logging.info( "Percentange of documents which have been improved to very high quality")
logging.info(str(len(X_array_re_ocred[X_array_re_ocred_pred==2])/len(re_OCRed_text_filename)))
print ("")

logging.info("Number of documents which have been improved to very high quality")
logging.info(str(len(X_array_re_ocred[X_array_re_ocred_pred==2])))
print ("")

logging.info( "Number of documents re-ocred")
logging.info("logging.info")
print ("")


# In[54]:

logging.info("Percentage of documents which has improved quality through re-ocring")
logging.info(str(num_imp/len(re_OCRed_text_filename)))
print ("")
logging.info("Backing up OCRed documents")


# In[55]:

from datetime import datetime


# In[56]:

re_ocr_backup_path=r'C:\Users\vbask\Documents\Darts_IP\Darts_IP\Factories\German_Factory\4_b_Re_OCRed_Backup'
directory=str(datetime.now()).split(" ")[0]+"_"+str(datetime.now()).split(" ")[1].split(":")[0]+"_"+str(datetime.now()).split(" ")[1].split(":")[1]
os.makedirs(os.path.join(re_ocr_backup_path,directory))


# In[57]:

for f in os.listdir(re_OCRed_text_path):
    shutil.copy(os.path.join(re_OCRed_text_path,f),os.path.join(re_ocr_backup_path,directory))


# In[58]:

logging.info("Finished the whole process")


# In[ ]:

shutil.copy(os.path.join(r'C:\Users\vbask\Documents\Darts_IP\Darts_IP\Factories\German_Factory_Code\germanFactory','mylog.log'),os.path.join(re_ocr_backup_path,directory))


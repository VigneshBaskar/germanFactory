
# coding: utf-8

# In[1]:

from __future__ import division
#matplotlib inline
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pymysql
import scipy.stats
from sklearn import linear_model
from sklearn import svm
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import os
import random
import warnings
import shutil
warnings.filterwarnings("ignore")


# In[4]:

import logging
logger = logging.getLogger()
os.chdir(r'C:\Users\vbask\Documents\Darts_IP\Darts_IP\Factories\German_Factory_Code\germanFactory')
fhandler = logging.FileHandler(filename='mylog.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

logging.info('Logging starts for source language detection')




# In[5]:

import io
from pdfminer.pdfinterp import PDFResourceManager,PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

def convert(fname, pages=None):
    if not pages:
        pagenums=set()
    else:
        pagenums=set(pages)
        
    output=io.BytesIO()
    manager=PDFResourceManager()
    converter=TextConverter(manager, output, codec='utf-8', laparams=LAParams())
    interpreter=PDFPageInterpreter(manager,converter)
    
    infile=file(fname,'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text=output.getvalue()
    output.close
    return text


# In[6]:

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


# In[7]:

logging.info("Loading the Latin Russian dictionary")
os.chdir(r'C:\Users\vbask\Documents\Darts_IP\Darts_IP\Dictionaries')
dictionary_words=pickle.load(open("latin_rus_dic.p","rb"))


# In[8]:

gen_OCRed_text_path="C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\German_Factory\\2_b_Poor_OCRed\\Batch_prior"
os.chdir(gen_OCRed_text_path)

gen_OCRed_text_filename=[f for f in os.listdir(gen_OCRed_text_path) if os.path.isfile(f) and os.path.splitext(f)[1]==".pdf"]
print gen_OCRed_text_filename,
print len(gen_OCRed_text_filename)


# In[10]:

logging.info("Extracting text from generically OCRed PDFs")
import time
time1=time.time()
import os

os.chdir(gen_OCRed_text_path)
gen_OCRed_text=[]
erroneous_fnames=[]
non_erroneous_fnames=[]

for fname in gen_OCRed_text_filename:
    try:
        logging.info("Successfully extracted text from "+fname)
        gen_OCRed_text.append(convert(fname).decode('utf-8','ignore'))
        non_erroneous_fnames.append(fname)

    except:
        print('Failed to extract text from '+fname)
        erroneous_fnames.append(fname)


time2=time.time()
logging.info("Time taken to extract text from "+str(len(gen_OCRed_text_filename))+" PDFs is "+str((time2-time1)/60))



# In[11]:

logging.info("Extracting features from generically OCRed text")
gen_OCRed_text_percent_tokens,gen_OCRed_text_len_tokens,gen_OCRed_text_processed=extract_quality_features(gen_OCRed_text,dictionary_words,0)
logging.info("Successfully extracted features from generically OCRed text")


# In[12]:

X_array_test=pickle.load(open("C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\German_Factory\\1_Poor_downloaded\\X_array_test.p","rb"))
test_text_filename=pickle.load(open("C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\German_Factory\\1_Poor_downloaded\\test_text_filename.p","rb"))


# In[13]:

poor_downloaded_path="C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\German_Factory\\1_Poor_downloaded\\Batch_prior"
poor_native_path="C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\German_Factory\\3_a_Poor_Native"
poor_img_path="C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\German_Factory\\2_a_Poor_img_layer\\Batch_prior"
poor_scanned_path="C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\German_Factory\\3_b_Poor_Scanned"


# In[14]:

logging.info("Deleting files from Native path")

gen_OCRed_text_filename=non_erroneous_fnames
import shutil
for root, dirs, files in os.walk(poor_native_path):
    for f in files:
        os.unlink(os.path.join(root, f))
        
logging.info("Deleting files from Scanned path")
for root, dirs, files in os.walk(poor_scanned_path):
    for f in files:
        os.unlink(os.path.join(root, f))
        


# In[15]:

logging.info("Detecting the source of the files")
Native_filename=[];Scanned_filename=[]
for index in range(len(gen_OCRed_text_filename)):
    if gen_OCRed_text_len_tokens[index] < 0.5*X_array_test[test_text_filename.index(gen_OCRed_text_filename[index])][1]:
        Native_filename.append(gen_OCRed_text_filename[index])
        shutil.copy(poor_downloaded_path+"\\"+gen_OCRed_text_filename[index],poor_native_path)
    else:
        Scanned_filename.append(gen_OCRed_text_filename[index])
        shutil.copy(poor_img_path+"\\"+gen_OCRed_text_filename[index],poor_scanned_path)
logging.info("Successfully detected the source of the files")


# In[16]:

logging.info("Detecting the language of the scanned pdfs")
import polyglot
from polyglot.detect import Detector
for index in range(len(Scanned_filename)):
    logging.info(Scanned_filename[index])
    try:
        for language in Detector(gen_OCRed_text[gen_OCRed_text_filename.index(Scanned_filename[index])],quiet=True).languages:
            logging.info(language)
    except:
        logging.info("Unable to detect language")
    print("\n")


# In[17]:

ocr_languages=["en","de","fr","es","it","ru","nl","pt","sv","tr","th","zh","ja","pl","ko"]


# In[19]:

detected_languages=[]
for index in range(len(Scanned_filename)):
    Confidence=[]
    temp_detected_languages=[]
    for language in Detector(gen_OCRed_text[gen_OCRed_text_filename.index(Scanned_filename[index])],quiet=True).languages:
        try:
            #print gen_OCRed_text_filename[gen_OCRed_text_filename.index(Scanned_filename[index])]
            if language.code in ocr_languages:
                Confidence.append(language.confidence)
                temp_detected_languages.append(language.code)
            if sum(Confidence)>85:
                break
        except:
            print "Do nothing"
    if list(set(temp_detected_languages[:2])-set(['en','zh','ru']))==[]:
        detected_languages.append([])
    else:
        detected_languages.append(temp_detected_languages[:2])
    



# In[23]:

for index in range(len(Scanned_filename)):
    logging.info(Scanned_filename[index]+" was detected to have the following langauges")
    logging.info(detected_languages[index])


# In[24]:

combinations=[]
for lang_1 in ocr_languages:
    for lang_2 in ocr_languages:
        if lang_1!=lang_2:
            combinations.append(lang_1+"_"+lang_2)
    ocr_languages=ocr_languages[1:]
ocr_languages=["en","de","fr","es","it","ru","nl","pt","sv","tr","th","zh","ja","pl","ko"]


# In[25]:

logging.info("Deleting the files in Re-OCR input")
for folder in combinations+ocr_languages:
    directory="C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\4_a_Re_OCRing_input\\"+folder
    for root, dirs, files in os.walk(directory):
        for f in files:
            os.unlink(os.path.join(root, f))
        
for folder in combinations+ocr_languages:
    directory="C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\4_a_Re_OCRing_input\\"+folder
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[26]:

logging.info("Deleting the files in Re-OCR output")

re_ocr_in_path="C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\4_a_Re_OCRing_input"
re_ocr_out_path="C:\\Users\\vbask\\Documents\\Darts_IP\\Darts_IP\\Factories\\German_Factory\\4_b_Re_OCRed"

for root, dirs, files in os.walk(re_ocr_out_path):
    for f in files:
        try:
            os.unlink(os.path.join(root, f))
        except:
            print" "


# In[27]:

logging.info("Transferring files to the corresponding directory")
for index in range(len(Scanned_filename)):
    index_len=len(detected_languages[index])
    if index_len==2:
        if os.path.exists(re_ocr_in_path+"\\"+detected_languages[index][0]+"_"+detected_languages[index][1]):
            shutil.copy(poor_scanned_path+"\\"+Scanned_filename[index],re_ocr_in_path+"\\"+detected_languages[index][0]+"_"+detected_languages[index][1])
        elif os.path.exists(re_ocr_in_path+"\\"+detected_languages[index][1]+"_"+detected_languages[index][0]):
            shutil.copy(poor_scanned_path+"\\"+Scanned_filename[index],re_ocr_in_path+"\\"+detected_languages[index][1]+"_"+detected_languages[index][0])
    if index_len==1:
        if os.path.exists(re_ocr_in_path+"\\"+detected_languages[index][0]):
            shutil.copy(poor_scanned_path+"\\"+Scanned_filename[index],re_ocr_in_path+"\\"+detected_languages[index][0])
    if index_len==0:
        shutil.copy(r"C:\Users\vbask\Documents\Darts_IP\Darts_IP\Factories\German_Factory\2_b_Poor_OCRed\Batch_prior"+"\\"+Scanned_filename[index],re_ocr_out_path)


# In[ ]:




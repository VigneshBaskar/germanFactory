
# coding: utf-8

# In[ ]:




# In[2]:

import subprocess32 as subprocess
import threading
import os


# In[8]:


import logging
logger = logging.getLogger()
os.chdir(r'C:\Users\vbask\Documents\Darts_IP\Darts_IP\Factories\German_Factory_Code\germanFactory')
fhandler = logging.FileHandler(filename='mylog.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

logging.info('Logging starts for Re-OCRing')




# In[4]:

def run_command_with_timeout(cmd):
    timeout_sec=200
    proc = subprocess.Popen(cmd)
    proc_thread = threading.Thread(target=proc.communicate)
    proc_thread.start()
    proc_thread.join(timeout_sec)
    if proc_thread.is_alive():
        subprocess.call(["taskkill","/f","/t","/im","FineReader.exe"])
        subprocess.call(["taskkill","/f","/t","/im","gswin64.exe"])
    return 0           


# In[5]:

ocr_languages=["en","de","fr","es","it","ru","nl","pt","sv","tr","th","zh","ja","pl","ko"]
combinations=[]
for lang_1 in ocr_languages:
    for lang_2 in ocr_languages:
        if lang_1!=lang_2:
            combinations.append(lang_1+"_"+lang_2)
    ocr_languages=ocr_languages[1:]
ocr_languages=["en","de","fr","es","it","ru","nl","pt","sv","tr","th","zh","ja","pl","ko"]


# In[6]:

re_ocr_in_path=r"C:\Users\vbask\Documents\Darts_IP\Darts_IP\Factories\4_a_Re_OCRing_input"
re_ocr_in_folder_names=os.listdir(re_ocr_in_path)
re_ocr_out_path=r'C:\Users\vbask\Documents\Darts_IP\Darts_IP\Factories\German_Factory\4_b_Re_OCRed'


# In[7]:

ocr_languages_full=["english","german","french","spanish","italian","russian","dutch","portuguesestandard","swedish","turkish","thai","chineseprc","japanese","polish","korean"]


# In[9]:

for re_ocr_in_folder_name in re_ocr_in_folder_names:
    if len(re_ocr_in_folder_name)>2:
        for re_ocr_fname in os.listdir(os.path.join(re_ocr_in_path,re_ocr_in_folder_name)):
            logging.info("ReOCRing "+re_ocr_fname)
            run_command_with_timeout(["C:\\Users\\vbask\\Documents\\ABBYY\\FineCmd.exe",os.path.join(os.path.join(re_ocr_in_path,re_ocr_in_folder_name),re_ocr_fname),"/lang",ocr_languages_full[ocr_languages.index(re_ocr_in_folder_name[0:2])],ocr_languages_full[ocr_languages.index(re_ocr_in_folder_name[3:5])],"/out",os.path.join(re_ocr_out_path,re_ocr_fname),"/quit"])
    if len(re_ocr_in_folder_name)==2:
        for re_ocr_fname in os.listdir(os.path.join(re_ocr_in_path,re_ocr_in_folder_name)):
            logging.info("ReOCRing "+re_ocr_fname)
            run_command_with_timeout(["C:\\Users\\vbask\\Documents\\ABBYY\\FineCmd.exe",os.path.join(os.path.join(re_ocr_in_path,re_ocr_in_folder_name),re_ocr_fname),"/lang",ocr_languages_full[ocr_languages.index(re_ocr_in_folder_name[0:2])],"/out",os.path.join(re_ocr_out_path,re_ocr_fname),"/quit"])            
    


# In[10]:

logging.info("Re-OCRing finished")


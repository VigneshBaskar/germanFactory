
# coding: utf-8

# In[1]:

import subprocess32 as subprocess
import threading
import os
import time


# In[24]:


import logging
logger = logging.getLogger()
os.chdir(r'C:\Users\vbask\Documents\Darts_IP\Darts_IP\Factories\German_Factory_Code\germanFactory')
fhandler = logging.FileHandler(filename='mylog.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

logging.info('Logging starts for Command line Access 1 files')




# In[11]:

def run_command_with_timeout(cmd):
    timeout_sec=200
    proc = subprocess.Popen(cmd)
    proc_thread = threading.Thread(target=proc.communicate)
    proc_thread.start()
    proc_thread.join(timeout_sec)
    if proc_thread.is_alive():
        logging.info("Killing tasks")
        subprocess.call(["taskkill","/f","/t","/im","FineReader.exe"])
        subprocess.call(["taskkill","/f","/t","/im","gswin64.exe"])
    return 0           


# In[8]:

prior_downloaded_path=r"C:\Users\vbask\Documents\Darts_IP\Darts_IP\Factories\German_Factory\1_Poor_downloaded\Batch_prior"
prior_downloaded_file_names=[f for f in os.listdir(prior_downloaded_path) if os.path.splitext(f)[1]==".pdf"]
prior_img_layer_path=r"C:\Users\vbask\Documents\Darts_IP\Darts_IP\Factories\German_Factory\2_a_Poor_img_layer\Batch_prior"
prior_gen_ocred_path=r"C:\Users\vbask\Documents\Darts_IP\Darts_IP\Factories\German_Factory\2_b_Poor_OCRed\Batch_prior"


# In[9]:

logging.info("Deleteing all the file in the priority Image layer only PDFs path")
for root, dirs, files in os.walk(prior_img_layer_path):
    for f in files:
        os.unlink(os.path.join(root, f))
        
logging.info("Deleteing all the file in the generic OCRed PDFs path")
for root, dirs, files in os.walk(prior_gen_ocred_path):
    for f in files:
        os.unlink(os.path.join(root, f))
        


# In[18]:

task=[]
for index in range(len(prior_downloaded_file_names)):
    task.append(["C:\\Program Files\\gs\\gs9.19\\bin\\gswin64","-o",os.path.join(prior_img_layer_path,prior_downloaded_file_names[index]),"-sDEVICE=pdfwrite","-dFILTERTEXT",os.path.join(prior_downloaded_path,prior_downloaded_file_names[index])])

from multiprocessing.dummy import Pool as ThreadPool 
s_time= time.time()
pool = ThreadPool(4) 
results = pool.map(run_command_with_timeout, task)
pool.close() 
pool.join() 
e_time=time.time()


# In[22]:

logging.info("The GS for "+ str(len(prior_downloaded_file_names))+" files ended in "+str((e_time-s_time)/60)+"minutes")
prior_img_layer_file_names=[f for f in os.listdir(prior_img_layer_path) if os.path.splitext(f)[1]==".pdf"]


# In[21]:

for index in range(len(prior_img_layer_file_names)):
    print index,
    logging.debug("Abbyy generic OCRing "+prior_img_layer_file_names[index] )
    run_command_with_timeout(["C:\\Users\\vbask\\Documents\\ABBYY\\FineCmd.exe",os.path.join(prior_img_layer_path,prior_img_layer_file_names[index]),"/lang","english","chineseprc","russian","/out",os.path.join(prior_gen_ocred_path,prior_img_layer_file_names[index]),"/quit"])
    time.sleep(1)


# In[23]:

logging.info("Generic OCRing finished")


# In[ ]:




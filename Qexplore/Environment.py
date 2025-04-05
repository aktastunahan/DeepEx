import cryptohash as chash
import random
import numpy as np
import pandas as pd
import wordninja
import nltk
#import gensim
import enchant
from sentence_transformers import SentenceTransformer
#import sister
from bs4 import BeautifulSoup
import string
from num2words import num2words
import matplotlib.pyplot as plt
import requests
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from prettytable import PrettyTable
import requests
import time
import exrex as ex
import js_regex
from sklearn.metrics.pairwise import cosine_similarity
import ast
import datetime
from spellchecker import SpellChecker
import os
import subprocess

class webEnv:
    
    def __init__(self,url,BaseURL="http://localhost/timeclock/",actionWait=0.5):
        self.url = url
        self.tags_to_find = ['input','button','a','select','option','audio','video', 'textarea',"submit","radio","checkbox","image"]
        self.tags_for_clickable = ['button','a','select','select2','audio','video',"submit","radio","checkbox","image"]
        self.tags_for_typable = ['input','textarea','search','password']
        self.tags_to_find = self.tags_for_clickable + self.tags_for_typable
        self.website  = webdriver.Chrome()
        self.website.get(url)
        self.website.execute_cdp_cmd('Network.setBlockedURLs', {"urls": ["www.saucelabs.com/"]})
        self.website.execute_cdp_cmd('Network.enable', {})
        self.original_window_id=self.website.current_window_handle
        self.datalabel = ['zipcode','city','streetname','secondaryaddress',
                'county','country','countrycode','state','stateabbr',
                'latitude','longitude','address','email','username',
                'password','sentence','word','paragraph','firstname',
                'lastname','fullname','age','phonenumber','date']
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        #self.embedding = self.model.encode("This is a test sentence.")
        #self.embedding = sister.MeanEmbedding(lang="en")
     
        self.edict = enchant.Dict('en_US')
        
        self.tagAttr = {'a':[''],'button':['value','name'],
                        'select':['name','class'],
                   'input':['placeholder','name', 'value']}
        self.prev_password = None
        self.BaseURL = BaseURL
        self.currentDepth=0
        self.actionWait = actionWait
        self.all_elements = self.website.find_elements("xpath", "//*")
        org_dir = os.getcwd()
        os.chdir("Data generator")
        subprocess.Popen(["npm", "start"], shell=True)
        os.chdir(org_dir)
        time.sleep(1)
    
    def get_all_elements(self):
        self.all_elements = self.website.find_elements("xpath", "//*")
        return self.all_elements

    def get_clickable_state_vector(self, N_max=64, pad_value=-1):
        """
        Extracts the state vector from the current web page.
        
        Args:
            N_max: Maximum length of the state vector.
            pad_value: Value to use for padding.

        Returns:
            A list of indices representing clickable elements, padded/truncated to N_max.
        """
        # Identify clickable elements
        clickable_indices = []
        for index, element in enumerate(self.all_elements):
            tag = element.tag_name.lower()
            type_attr = element.get_attribute("type")
            is_text=( type_attr == "text" 
                        or type_attr == "" 
                        or type_attr == "password"
                        or type_attr == "email"
                        or type_attr == None)
            
            is_clickable = (tag in self.tags_for_clickable or (tag == "input" and not is_text) or 
                element.get_attribute("onclick") is not None  # Check inline JS click events
            )
            
            is_typeable = (
                (tag == "textarea") or (tag == "input" and is_text) 
            )
            is_formfill = (tag == "form" or tag == "fieldset")
            if is_clickable or is_typeable or is_formfill:
                clickable_indices.append(index)

        # Pad or truncate to ensure fixed length
        if len(clickable_indices) < N_max:
            clickable_indices.extend([pad_value] * (N_max - len(clickable_indices)))
        else:
            clickable_indices = clickable_indices[:N_max]
        
        return clickable_indices
    
    def get_random_string(self,length):
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    def getvectors(self,sentences):
        vector = self.model.encode(sentences)
        return vector
    
    def getsimilarity(self, feature_vec_1, feature_vec_2):    
        return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]
    
    def getdistance(self,a,b):
        return np.linalg.norm(a-b)
    
    def getSentence(self,html):
        soup = BeautifulSoup(html,'lxml')
        sentence = ""
        table = str.maketrans('', '', string.punctuation)
        for tag in ['input','button','a','select']:
            for t in soup.findAll(tag):
                att = t.attrs
                for chose in self.tagAttr[tag]:
                    try:
                        v = att[chose]
                        if type(v)==list:
                            sentence+=" ".join(v)+" "
                        else:
                            sentence+=' '+v
                    except:
                        continue
                
                sentence+=t.text
                #sentence = t.name+" "+sentence
        #print("before: ",sentence)
        #or cls in self.bootclasses:
        #   sentence = sentence.strip().replace(cls,'')
        #or cls in self.stopwords:
        #   sentence = sentence.strip().replace(cls,'')
            
        sentence = sentence.strip().translate(table)
        sentence = sentence.lower().replace('lastname','last-name')
        sentence = sentence.replace('firstname','first-name')
        sentence = sentence.replace('username','user-name')
        sentence = sentence.replace('userid','user-name')
        sentence = sentence.replace('enddate','end-date')
        sentence = sentence.replace('startdate','start-date')
        sentence = sentence.replace('cnic','identitynumber')
        #print("after class: ",sentence)
        sentence_new = ""
        for num in wordninja.split(sentence):
            word = ''.join([i for i in num if not i.isdigit()])
            try:
                #if word in self.spell and len(word)>1:
                if self.edict.check(word) and len(word)>1:
                    sentence_new+=word+" "
                    #print(word)
            except:
                pass
        #print("after ninja: ",sentence_new)
        return " ".join(list(set(sentence_new.split(" "))))
    
    #This method execute each element of the DOM depending on the type of element
    
    def click(self,elem):
        try:
            elem.click()
            time.sleep(self.actionWait)
            return 1
        except:
            return 0        
    
    def write(self,elem,login_url,username,password,email):
        html = elem.get_attribute("outerHTML")
        sentence = self.getSentence(html)
        
        if sentence!="":
            if "user" in sentence and "name" in sentence: 
                mostsim = "username"
            else:
                v_sentence = self.getvectors(sentence)
                #print("***********************")
                simL = []
                for x in self.datalabel:
                    x_vector = self.getvectors(x)
                    sim = self.getsimilarity(x_vector,v_sentence)
                    simL.append(sim)
                    #print(x+" is "+str(sim)+" similar to '"+sentence+"'")
                mostsim = self.datalabel[np.argmax(simL)]
            #print("'"+sentence+"' is most similar to "+mostsim)
            PARAMS = {'value':mostsim}
            r = requests.get(url = "http://localhost:3000/", params = PARAMS)
            d = ast.literal_eval(r.text.replace("`",""))
            if mostsim=='date':
                date = d["'d'"][0].split("T")[0]
                date = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y')
                d["'d'"][0]=date
            if mostsim=="paragraph" or mostsim=="word":
                d["'d'"][0] = "abcd123456"
            if mostsim=="username": # or mostsim=="lastname"
                if self.website.current_url in login_url:
                    if username!=None:
                        d["'d'"][0] = username
                else:
                    if username!=None:
                        d["'d'"][0] = username
                    print(self.website.current_url,login_url,self.website.current_url in login_url)
            if mostsim=="email":
                if self.website.current_url in login_url:
                    if email!=None:
                        d["'d'"][0] = email
                else:
                    print(self.website.current_url,login_url,self.website.current_url in login_url)
            if mostsim=="password":
                if self.website.current_url in login_url:
                    if password!=None:
                        d["'d'"][0] = password
                else:
                    if password!=None:
                        d["'d'"][0] = password
                        
                    if self.prev_password!=None:
                        d["'d'"][0] = self.prev_password
                    else:
                        self.prev_password = d["'d'"][0]

            #print("Generated: ",d["'d'"][0]," mostsim=",mostsim)
            try:
                if elem.get_attribute("value")=="" or elem.get_attribute("value")==None:
                    if elem.get_attribute("value")!=d["'d'"][0]:
                        elem.clear()
                        elem.send_keys(d["'d'"][0])
                return 1
            except:
                return 0
        else:
            PARAMS = {'value':'word'}
            r = requests.get(url = "http://localhost:3000/", params = PARAMS)
            d = ast.literal_eval(r.text.replace("`",""))
            try:
                if elem.get_attribute("value")=="" or elem.get_attribute("value")==None:
                    if elem.get_attribute("value")!=d["'d'"][0]:
                        elem.clear()
                        elem.send_keys(d["'d'"][0])
                return 1
                #elem.send_keys(d["'d'"][0])
                #return 1
            except:
                return 0

    def checkDone(self,depth):
        if self.currentDepth>=depth:
            return True
        else:
            return False
    
    
    def step(self,elem,login_url="",username=None,password=None,depth=4,email=None):
        
        clickable = ["a","button","submit","select","radio","checkbox","image","select2"]
        writable = ["input","text","password","search"]
        formfill = ["form","fieldset"]
        status = 0
        
        if elem.tag_name in clickable or elem.get_attribute('Type') in clickable:
            if elem.tag_name=="select":
                try:
                    select = Select(elem)
                    option = random.choice(select.options)
                    status = self.click(option)
                except:
                    status = 0
            else:
                status = self.click(elem)
                
            if status:
                self.currentDepth+=1
                
        elif elem.tag_name in writable or elem.get_attribute('Type') in writable:
            status = self.write(elem,login_url,username,password,email)
            if status:
                self.currentDepth+=1
        elif elem.tag_name in formfill:
            children = elem.find_elements("xpath", "./*")
            status = 0
            done = False
            for child in children:
                status_step, done_step = self.step(child, login_url,username,password,depth,email)
                done |= done_step
                if status_step != 0:
                    status = status_step
            return status, done
        else:
            return 0,self.checkDone(depth)
        
        return status,self.checkDone(depth)

    def getUrl(self):
        return self.website.current_url
    
    def close_tabs(self):
        if len(self.website.window_handles) > 1:
            for window in self.website.window_handles:
                if window != self.original_window_id:
                    self.website.switch_to.window(window)
                    self.website.close()
                    self.website.switch_to.window(self.original_window_id)
        if "saucelabs" in self.website.current_url:
            self.website.back()

    def draw_square(self,element):
         self.website.execute_script("""
                const el = arguments[0];
                const rect = el.getBoundingClientRect();
                const box = document.createElement('div');
                box.style.position = 'fixed';
                box.style.top = rect.top + 'px';
                box.style.left = rect.left + 'px';
                box.style.width = rect.width + 'px';
                box.style.height = rect.height + 'px';
                box.style.border = '3px solid red';
                box.style.zIndex = 9999;
                box.style.pointerEvents = 'none';  // Don't block mouse events
                box.setAttribute('data-selenium-highlight', 'true');  // 🔥 Add this line
                document.body.appendChild(box);
            """, element)

    def remove_drawn_square(self):
        self.website.execute_script("""
            document.querySelectorAll('[data-selenium-highlight="true"]').forEach(el => el.remove());
        """)
        
    def reset(self,curl=""):
        if curl=="":
            self.website.get(self.url)
            self.currentDepth=0
            self.prev_password = None
        else:
            self.website.get(curl)
            #self.currentDepth=0
            #self.prev_password = None
    
    def close(self):
        try:
            self.website.close()
            self.website.close()
        except:
            pass
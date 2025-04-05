#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding
from tensorflow.keras.optimizers import Adam

from Memory import Memory

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()

def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3


# In[2]:



import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
from IPython.display import clear_output


if "../" not in sys.path:
    sys.path.append("../") 

from collections import defaultdict

matplotlib.style.use('ggplot')


# In[3]:


from tqdm import tqdm


# In[4]:


from Environment import *


# In[5]:


import pandas as pd
import random
    
class DeepEx: 
    _learn_rate = None
    _discount_factor = None

    def __init__(self,
                 state_size = 100,
                 action_size = 100,
                 lr=0.001,
                 discount_factor = 0.9,
                 epsilon = 1.0,
                 memory = None):
        # Initialize state, action sizes, memory, exploration parameters, and neural network (Algorithm 1: Initialization)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = memory

        self.epsilon_min = 0.2  # Minimum exploration rate (Algorithm 1: εmin)
        self.epsilon_decay = 0.995  # Exploration decay rate

        self.model = self.build_model(lr)

        # Save the parameters
        self._learn_rate = lr
        self._discount_factor = discount_factor
        self.epsilon = epsilon

    def build_model(self, lr):
            # Construct the neural network (DQN)
            model = Sequential()
            model.add(Embedding(1000, 32, input_length=self.state_size))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr))
            return model
    
    def replay(self, batch_size=32):
        # Replay experiences to train the network
        minibatch = self.memory.sample(batch_size)
        if minibatch == None:
            return
        for state, action, reward, next_state in minibatch:
            act_idx = state.index(action)
            target = reward + self._discount_factor * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][act_idx] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(self, prev_state, next_state):
        # Calculate reward based on new elements discovered 
        new_elements = np.setdiff1d(list(filter((-1).__ne__, next_state)), list(filter((-1).__ne__, prev_state)))
        reward = len(new_elements) / len(next_state)
        print(f"reward: {reward}")
        return reward
          
    def get_best_action(self, state, policy=True):
        num_clickable_actions = sum(1 for x in state[:self.state_size] if x != -1)
        if num_clickable_actions < 2:
            return -1
        if policy:
            # Select action based on epsilon-greedy policy
            if np.random.rand() <= self.epsilon:
                print(f"policy:True random")
                return state[random.randrange(num_clickable_actions)]

            q_values = self.model.predict(np.array([state]), verbose=0)[0]
            argmax = np.argmax(q_values)
            print(f"policy:True argmax:{argmax} is state[{argmax}]: {argmax < num_clickable_actions}")
            return state[argmax] if argmax < num_clickable_actions else state[random.randrange(num_clickable_actions)]
        else:
            q_values = self.model.predict(np.array([state]), verbose=0)[0]
            argmax = np.argmax(q_values)
            print(f"policy:False argmax:{argmax} is state[{argmax}]: {argmax < num_clickable_actions}")
            return state[argmax] if argmax < num_clickable_actions else state[random.randrange(num_clickable_actions)]    

    def remember(self, state, action, reward, next_state):
        self.memory.remember(state, action, reward, next_state)

from apted import APTED
from apted.helpers import Tree
import pandas as pd
import numpy as np
import glob, os
from tqdm import tqdm
import itertools
from bs4 import BeautifulSoup
import bs4
#from collections import defaultdict

from collections import OrderedDict
from collections.abc import Callable  

import hashlib
import base64

class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))



def defaultVal():
    return [[],0]

def makeTree(S):
    S = S.strip()
    S = S.replace("\n","")
    #S = S.replace(" ","")
    S = S.replace("\t","")
    S = S.replace("\r","")
    soup = BeautifulSoup(S, "html.parser")
    return soup

def recursiveChildBfs(bs):
    root = bs
    stack = [root]
    count=0
    parrent = [None]
    while len(stack) != 0:
        node = stack.pop(0)
        pnode = parrent.pop(0)
        if node is not bs:
            if node.name!=None:
                yield node.name+"~"+str(count),pnode
            else:
                yield node.name,pnode
        if hasattr(node, 'children'):
            for child in node.children:
                stack.append(child)
                parrent.append(node.name+"~"+str(count))
        count+=1

def visit(tagdict,c,tree):
    tree+="{"
    tree+=c.split("~")[0]
    for i in tagdict[c][0]:
        tree = visit(tagdict,i,tree)
        tree+="}"
    return tree        

def generateTree(S):
    html = makeTree(S)
    tagdict = DefaultOrderedDict(defaultVal)
    for c,p in recursiveChildBfs(html):
        if c!=None:
            tagdict[p][0].append(c)
            tagdict[p][1]+=1


    tree = "{"
    for x,y in zip(list(tagdict.keys())[1::],list(tagdict.values())[1::]):
        tree+=x.split("~")[0]
        for c in y[0]:
            #tree+="{"
            #tree+=c
            tree = visit(tagdict,c,tree)
            tree+="}"
        tree+="}"
        break
    nNodes = 0
    for x in tagdict.keys():
        nNodes+=tagdict[x][1]
    return tree,nNodes


# In[6]:


import time
import numpy as np
import pandas as pd
import os
import shutil
from collections import defaultdict
import pickle
import json
from threading import Thread

CLOSE=False

def save_stateMap(obj, name):
    with open(name,"w") as dd:
        dd.write(json.dumps(obj))

def load_stateMap(name):
    with open(name) as json_file:
        data = defaultdict(factory)
        data2 = json.load(json_file)
        data.update(data2)
    return data

def factory():
    return {"src":"","edges":[],"url":"","start":0}

def makeGraph(stateMap,output):
    statesfile = [file.split("/")[-1].split(".html")[0] for file in glob.glob(output+"/*.html")]
    with open(os.path.join(output,"data.js"),"w") as jsonwriter:
        C = []
        for x in stateMap.keys():
            stateMap[x]["edges"] = [{"action":y["action"],"state":y["state"]} for y in stateMap[x]["edges"] if any(y["state"] in s for s in statesfile)]
            C.append(stateMap[x])
        C.sort(key=lambda x:x["start"],reverse=True)
        jsonwriter.write("let data = ")
        jsonwriter.write(json.dumps(C))

def hash_state(state):
    # Slice list up to -1
    try:
        index = state.index(-1)
        sublist = state[:index]
    except ValueError:
        sublist = state  # If -1 not in list, use the whole list

    # Convert to bytes and hash
    bytes_data = bytes(sublist)
    
    hash_bytes = hashlib.sha256(bytes_data).digest()
    short_str = base64.urlsafe_b64encode(hash_bytes).decode('utf-8')
    return chash.md5(short_str)


def deep_ex_learning(env, num_episodes,sleep=0,matrix=None,statemap=None,_epsilon=0.2,
               onlyperform=False,output="./Q_Result",timebound=True,activity_time=None,
              login_urls=[],username="",password="",depth=100):
    global CLOSE
    #stats = EpisodeStats(
    #    episode_lengths=np.zeros(num_episodes),
    #    episode_rewards=np.zeros(num_episodes))    
    
    env.reset()

    
    memory = Memory()
    deepEx = DeepEx(memory=memory)

        #onlyperform = True
    #display(Qlearn._qmatrix)
    rewardlist = ["http://192.168.1.68/timeclock/admin/groupdelete.php","http://192.168.1.68/timeclock/admin/index.php"]
    curl = ""
    
    #-=-=-=-=makedir=-=-=-=
    stateMap = defaultdict(factory)
    if os.path.exists(output):
        if statemap==None:
            shutil.rmtree(output)
            os.mkdir(output)
        else:
            stateMap = statemap
    else:
        os.mkdir(output)
    for code in os.listdir("./graphView/"):
        shutil.copyfile(os.path.join("./graphView/",code),os.path.join(output,code))
    #-=-=-==-=-=-=-=-=-=-=-
    if timebound:
        def some_task():
            global CLOSE
            time.sleep(activity_time)
            CLOSE=True
        t = Thread(target=some_task)
        t.start()
    
    
    all_elements = env.get_all_elements()
    state = env.get_clickable_state_vector()
    startstate = state
    i_episode=0
    while(True):
        print("EPISODE= ",i_episode)
        if timebound:
            if CLOSE:
                break
        else:
            if i_episode>num_episodes:
                break
              
        #Qlearn._qmatrix.to_csv("./Q-table")
        save_stateMap(stateMap,"Q.map")
        env.reset(curl)
        curl = ""
            
        all_elements = env.get_all_elements()
        state = env.get_clickable_state_vector()
        prev_action = ""
        urlbefore = ""
        startstate = hash_state(state)
        for t in itertools.count():
            start = time.time()
            urlbefore = env.getUrl()
            
            #=-=--=-=-STATE GRAPH-=-=-=-=-
            state_str = hash_state(state)
            env.website.save_screenshot(os.path.join(output,state_str+".png"))
            with open(os.path.join(output,state_str+".html"),"w") as htmlwriter:
                htmlwriter.write(env.website.page_source)
            #-=--=-=-=--=-=-=-=-=-=-=-=-=
            
            action = deepEx.get_best_action(state)
            
            if action == -1:
                env.reset()
                break
            env.draw_square(all_elements[action])
            time.sleep(0.33
                       )
            html = all_elements[action].get_attribute("outerHTML")
            print(f"selected action html: {html[:100]}")
            env.remove_drawn_square()
            status,done = env.step(all_elements[action],login_url=login_urls,
                                       username=username,password=password,
                                       depth=depth)
            env.close_tabs()
            all_elements = env.get_all_elements()
            next_state = env.get_clickable_state_vector()
            
            reward = deepEx.calculate_reward(state, next_state)
            deepEx.remember(state, action, reward, next_state)
            

             #-=-=-=-=makingGraph=-=-=-=-
            stateMap[state_str]["src"]=state_str
            stateMap[state_str]["edges"].append({"action":action,"state":hash_state(next_state)})
            if state_str==startstate:
                stateMap[state_str]["start"] = 1
            stateMap[state_str]["url"] = urlbefore
                #-==-=-=-=-=-=-=-=-=-=-=-=-=
            state = next_state
            prev_action = action
            end = time.time()
            print(f"step {t} time: {end-start} sec.")
            if t % 10 == 0:
                deepEx.replay()
                makeGraph(stateMap,output)
            
        #print(actionlist)
        i_episode+=1
    
    #return stats,Qlearn,stateMap
    return 0,deepEx,stateMap


# In[7]:


import argparse
from gooey import Gooey

#@Gooey
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="The home page url of the website", default="https://www.saucedemo.com/")
    parser.add_argument("--login_urls", help="The login page url of the website, add comma incase of multiple", default="https://www.saucedemo.com/")
    parser.add_argument("--username", help="The username",default="standard_user")
    parser.add_argument("--password", help="The password", default="secret_sauce")
    parser.add_argument("--baseurl", help="The base url to check for out of scope websites(without Tailing /), Default is same as home url",default="https://www.saucedemo.com/")
    parser.add_argument("--action_wait", help="Action weight time in seconds, default is 0.5",default=0.5,type=float)
    parser.add_argument("--episodes", help="Number of episodes to run, default in 2",default=10,type=int)
    parser.add_argument("--matrix", help="path of matrix to resume, default is None",default=None)
    parser.add_argument("--stateMap", help="path of statemap to resume, default is None",default=None)
    parser.add_argument("--timebound", help="To use time instead of episodes, default is False",action="store_true")
    parser.add_argument("--activity_time", help="max time to run the activity, default is 0",default=0,type=int)
    parser.add_argument("--depth", help="max valid actions in one episode, default is 100",default=100,type=int)
    
    args = parser.parse_args()
    
    
    #env = webEnv(url="http://192.168.1.68/timeclock/",BaseURL="http://192.168.1.68/timeclock",actionWait=0.5)
    login_urls = args.login_urls.split(",")
    base = "https://www.saucedemo.com/"
    if args.baseurl:
        base = args.baseurl
    else:
        base = args.url[::-1]
        
    env = webEnv(url=args.url,BaseURL=base,actionWait=args.action_wait)
    
    #stats,matrix,stateMap=q_learning(env, 2,timebound=True,activity_time=60)
    stats,matrix,stateMap=deep_ex_learning(env, args.episodes,timebound=args.timebound,activity_time=args.activity_time,
                                    matrix=args.matrix,statemap=args.stateMap,
                                    login_urls=login_urls,username=args.username,password=args.password,depth=args.depth)


# In[8]:

if __name__ == "__main__":
    main()


# In[ ]:





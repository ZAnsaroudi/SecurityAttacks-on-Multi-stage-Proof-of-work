import sys
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d
import math
from mpmath import *
mp.dps = 20


# block level selfish minning on Multi-stage proof of work blockchain:@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Selfish_BlockMining:
    # def __init__(self, nb_simulations, nb_stages alpha):
    def __init__(self, **d):
        # Setted Parameters
        self.__alpha = d['alpha']
        self.nb_stage = d['nb_stage']
        self.__nb_simulations = d['nb_simulations']
        self.__privateChain = 0  # length of private chain RESET at each validation
        self.__honestsValidBlocks = 0
        self.__selfishValidBlocks = 0
        self.__counter = 1
        # For results
        self.__revenue = None
        self.__orphanBlocks = 0
        self.__totalMinedBlocks = 0
        # For finding probabilities
        self.Rn=self.nb_stage  # Remaining number of stages
        self.expected=0
        self.devision=0
        self.conselfishprob=0
        self.movedback=False
        self.movedforward=False

    def write_file(self):
        stats_result = [self.__alpha, self.__nb_simulations, \
                        self.__honestsValidBlocks, \
                        self.__selfishValidBlocks, \
                        self.__revenue, self.__orphanBlocks, self.__totalMinedBlocks]

    def binom(self,n, k):
        return math.factorial(n)/( math.factorial(k) * math.factorial(n - k))

# For finding remaining number of stages
    def __residue(self,i,p1):
        switcher = {
            1: self.__residue_one(p1),
            2: self.__residue_two(p1),
            3: self.__residue_three(p1)
        }
        return switcher.get(i, "nothing")

    def __residue_one(self,p1):
        ### moved forward
        #Rn=k−stageh−1 or k-stageh, once the selfish has solved k stages, how many is solved by the honest?
        self.expected=0
        self.devision=0
        for n in range(self.Rn):
            self.expected = self.expected + (n*(p1 ** self.nb_stage) * ((1 - p1) ** (n)) * self.binom(
                n + self.nb_stage - 1, self.nb_stage - 1))

        for m in range(self.Rn):
            self.devision = self.devision + ((p1 ** self.nb_stage) * ((1 - p1) ** (m)) * self.binom(
                m + self.nb_stage - 1, self.nb_stage - 1))

        if self.__privateChain==1: self.Rn= self.nb_stage-int(self.expected/self.devision)-1
        else: self.Rn= self.nb_stage-int(self.expected/self.devision) #where the selfish pool unpublished block is greater than 1
        return self.Rn

    def __residue_two(self,p1):
        ### moved back
        #Rn=k−stages, once the honest has solved k-1 or k stages, how many has solved by the selfish
        #print("remaing2--------------------------------------------------------------", self.Rn)
        self.expected=0
        self.devision=0
        No=0 # number of stages solved by the honest
        if self.__privateChain>1: No=self.nb_stage
        else: No=self.nb_stage-1
        for n in range(No):
            self.expected = self.expected + (n* (p1 ** self.Rn) * ((1 - p1) ** (n)) * self.binom(
                n + self.Rn - 1, self.Rn - 1))
        for m in range(No):
             self.devision = self.devision + ((p1 ** self.Rn) * ((1 - p1) ** (m)) * self.binom(
                 m + self.Rn - 1, self.Rn - 1))

        self.Rn= self.nb_stage-int(self.expected/self.devision)
        return self.Rn

    def __residue_three(self,p1):
        ### in state 0 , k-stage_h
        #print("remaing3----------------------------------------------------------------------", self.Rn)
        self.expected=0
        self.devision=0
        for n in range(self.nb_stage):
            #print("n",n)
            #print("sigmaex", self.expected)
            self.expected = self.expected + (n* (p1 ** self.Rn) * ((1 - p1) ** (n)) * self.binom(
                n + self.Rn - 1, self.Rn - 1))
        for m in range(self.nb_stage):
            #print("sigmade", self.devision)
            self.devision = self.devision + (p1 ** self.Rn) * ((1 - p1) ** (m)) * self.binom(
                m + self.Rn - 1, self.Rn - 1)
        #print("************************")
        #print("p1",p1)
        #print("PREVIOUS",self.Rn)
        self.Rn= self.nb_stage-int(self.expected/self.devision)
        #print(self.expected)
        #print("UPDATED3", self.Rn)
        return self.Rn

# For finding mining probability of pools
    def __prob_one(self, p1):
        ### moved forward
        self.conselfishprob = 0
        for n in range(self.Rn):
            self.conselfishprob = self.conselfishprob + (p1 ** self.nb_stage) * ((1 - p1) ** (n)) * self.binom(
                n + self.nb_stage - 1, self.nb_stage - 1)
        return self.conselfishprob

    def __prob_two(self, p1):
        ### moved back
        self.conselfishprob = 0
        for n in range(self.nb_stage - 1):
         self.conselfishprob = self.conselfishprob + (p1 ** self.Rn) * ((1 - p1) ** (n)) * self.binom(
                n + self.Rn - 1, self.Rn - 1)
        return self.conselfishprob

    def __prob_three(self, p1):
        ### in state 0
        self.conselfishprob=0
        for n in range(self.nb_stage):
            self.conselfishprob = self.conselfishprob + (p1 ** self.Rn) * ((1 - p1) ** (n)) * self.binom(
                    n + self.Rn - 1, self.Rn - 1)
        return self.conselfishprob

    def __probability(self,i):
        p1 = self.__alpha
        #prob is honest mining probability
        prob=0
        for n in range(self.nb_stage):
            prob = prob + (p1 ** self.nb_stage) * ((1 - p1) ** (n)) * self.binom(
                n + self.nb_stage - 1, self.nb_stage - 1)

        switcher = {
            #1: move forward
            1: self.__prob_one(p1),
            #2:moved back
            2: self.__prob_two(p1),
            #3: in state 0
            3: self.__prob_three(p1)
                    }
        return switcher.get(i, "invalid"),prob

    def __simprobability(self):
        prob=0
        if self.movedback:
            prob = self.__probability(2)[0]
            #print("prob_back",prob)
        # machine has moved forward
        elif self.movedforward:
            prob = self.__probability(1)[0]
            #print("prob_forward",prob)
        else:
            # machine has started or looped in state 0
            prob = self.__probability(3)[0]
            #print("prob_loop",prob)
        return prob

# For Simulation of Selfish mining for nb_simulations
    def Simulate(self):
        while (self.__counter <= self.__nb_simulations):
            s = np.random.uniform(0, 1)  # random number for each simulation
            if s <= self.__simprobability():
                print("mined by selfish *************************************************************")
                self.__privateChain += 1
                if self.__counter==1:
                    # is in state i==1 #Rn=k−stageh−1 once the selfish has solved k stages, how many is solved by the honest?
                    self.__residue(1,self.__alpha)

                elif self.movedback:
                    # the selfish pool has returned to previous state in the last transition of state machine
                    self.__residue(2, self.__alpha)
                    self.movedback=False

                elif self.movedforward:
                    self.__residue(1, self.__alpha)

                else:
                    self.__residue(3,self.__alpha)

                self.movedforward=True

            else:
                print("mined by honest **************************************************************")
                if self.__privateChain>=1:
                       self.__selfishValidBlocks += 1
                       #self.__honestsValidBlocks +=1
                       self.__privateChain -= 1
                       self.__residue(2,self.__alpha)
                       self.movedback=True
                       self.movedforward = False
                else:
                       self.__honestsValidBlocks += 1
                       self.Rn= self.nb_stage
                       self.movedforward = False
                       self.movedback = False

            self.__counter += 1
        while(self.__privateChain>0): # Publishing private chain if not empty when total nb of simulations reached
            self.__selfishValidBlocks += 1
            self.__privateChain-= 1

        self.actualize_results()
        if self.__revenue==None:
            self.__revenue=0
        return (self.__revenue, self.__probability(0)[1]*100)

# For rewards
    def actualize_results(self):
        # Total Blocks Mined
        self.__totalMinedBlocks = self.__honestsValidBlocks + self.__selfishValidBlocks
        # Orphan Blocks
        self.__orphanBlocks = self.__nb_simulations - self.__totalMinedBlocks
        # Revenue
        if self.__honestsValidBlocks or self.__selfishValidBlocks:
            self.__revenue = round(self.__selfishValidBlocks/self.__totalMinedBlocks, 3)

###########################################################
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

if len(sys.argv) == 4:
    dico = {'nb_simulations': int(sys.argv[1]), 'nb_stage':float(sys.argv[2]),'alpha': float(sys.argv[3]), 'gamma': float(sys.argv[4])}
    new = Selfish_BlockMining(**dico)
    new.Simulate

if len(sys.argv) == 1:
    ### TO SAVE MULTIPLE VALUES IN FILE ###
    start = time.time()
    alphas = list(i / 100 for i in range(1, 51, 1))  # 50 => 0, 0.5, 0.01
    alphas_stage = list(i / 100 for i in range(50, 100, 1))  # 50 => 0, 0.5, 0.01
    stage_No=[2,7,15,25]
    block_No = 1000
    count = 0  #pourcentage done
    container_block = []
    honestprob=[]
    plot = []

for stg in stage_No:
 #print("stage---------------------------------------------------------------------: ", stg)
 for alpha in alphas:
    #print("alpha------------------------------------------------------------------: ", alpha)
    new = Selfish_BlockMining(**{'nb_simulations': block_No, 'nb_stage': stg, 'alpha': alpha})
    container_block.append(new.Simulate()[0]) # append revenue
    honestprob.append(new.Simulate()[1]/100)
    new.write_file()
    duration = time.time() - start

y_elem = list(divide_chunks(container_block, len(alphas)))
hprob = list(divide_chunks(honestprob, len(alphas)))
#xnew = np.linspace(min(alphas), max(alphas), 5)

for j in range(len(y_elem)):
     #if (j == 0):
     ysmoothed = gaussian_filter1d(y_elem[j], sigma=2.5)
     #elif (j == 1):
         #ysmoothed = gaussian_filter1d(y_elem[j], sigma=2.5)
     #elif(j == 2):
         #ysmoothed = gaussian_filter1d(y_elem[j], sigma=2.5)
     #else:
         #ysmoothed = gaussian_filter1d(y_elem[j], sigma=2.5)
    #power_smooth
     plt.plot(alphas, ysmoothed , label='Selfish Mining, ' + str(stage_No[j]) +" stages")
     plt.legend()
plt.xlabel('Pool size')
plt.ylabel('Pool revenue')

for i in range(len(hprob)):
    #xnew = np.linspace(min(alphas), max(alphas),5)  # 5 represents number of points to make between T.min and T.max
    #spl = make_interp_spline(alphas, hprob[i], k=3)  # BSpline object
    #power_smooth = spl(xnew)
    if i == 0:
        label = 'Honest Mining'
    else:
        label = ''
    #plt.plot(xnew, power_smooth, ':k', label=label)
    plt.plot(alphas, hprob[i], ':k', label=label)
    plt.legend()

plt.show()


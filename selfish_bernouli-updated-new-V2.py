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
        self.__privateChain = 0
        self.publicchain = 0
        self.__honestsValidBlocks = 0 #number of blocks published by the honest pool
        self.__selfishValidBlocks = 0 #number of blocks published by the selfish pool
        self.__counter =1 # counts the number of simulations
        # For results
        self.__revenue = None #the selfish pool revenue
        self.__orphanBlocks = 0
        self.__totalMinedBlocks = 0
        # For finding probabilities
        self.selfishprob=0 #selfish pool probability for each event of state machine
        self.R = self.nb_stage  # Remaining number of stages
        # For finding expected number of stages expected=numerator/division
        self.numerator = 0
        self.division = 0
        # For tracking state machine
        self.movedback=False
        self.movedforward=False
        self.sameevent=False #indicates that if two same event(like moving forward, moving back are happened in order)
        self.currentstate=0
        self.previousstate=0
        self.stage_sel=0
        self.stage_hon=0


    def write_file(self):
        stats_result = [self.__alpha, self.__nb_simulations, \
                        self.__honestsValidBlocks, \
                        self.__selfishValidBlocks, \
                        self.__revenue, self.__orphanBlocks, self.__totalMinedBlocks]

    def binom(self,n, k):
        return math.factorial(n)/( math.factorial(k) * math.factorial(n - k))

# For finding remaining number of stages
    def __expectedstages(self,p1, p1param, otherparam):
       #The honest pool has completedat mostR−1stages in the meantime
        self.numerator=0
        self.division=0
        for n in range(otherparam):
            self.numerator = self.numerator + (n*(p1 ** p1param) * ((1 - p1) ** (n)) * self.binom(
                n + p1param - 1, p1param - 1))
        for m in range(otherparam):
            self.division = self.division + ((p1 ** p1param) * ((1 - p1) ** (m)) * self.binom(
                m + p1param - 1, p1param - 1))

        self.expected= int(self.numerator/self.division)
        return self.expected

# For finding mining probability of pools
    def __prob_moveforward(self, p1):
        ### machine moved forward, selfish pool has to solve k stages where honest solves R stages
        self.selfishprob = 0
        for n in range(self.R):
            self.selfishprob = self.selfishprob + (p1 ** self.nb_stage) * ((1 - p1) ** (n)) * self.binom(
                n + self.nb_stage - 1, self.nb_stage - 1)
        return self.selfishprob

    def __prob_moveback(self, p1):
        ### machine moved back, selfish solves his remaining number of stages untill honest solve his stages
        # where state i>1 selfish needs to finish his remaining stages where honest
        self.selfishprob = 0
        if self.currentstate==1: No=self.nb_stage-1
        else: No=self.nb_stage
        for n in range(No):
         self.selfishprob = self.selfishprob + (p1 ** self.R) * ((1 - p1) ** (n)) * self.binom(
                n + self.R - 1, self.R - 1)
        return self.selfishprob

    def __prob_state0(self, p1):
        ### in state 0
        self.selfishprob=0
        ### compute Mb+Mc
        for n in range(self.nb_stage):
            self.selfishprob = self.selfishprob + (p1 ** self.R) * ((1 - p1) ** (n)) * self.binom(
                    n + self.R - 1, self.R - 1)
        return self.selfishprob

    def __probability(self):
        #i
        p1 = self.__alpha
        #prob is honest mining probability
        prob=0
        for n in range(self.nb_stage):
            prob = prob + (p1 ** self.nb_stage) * ((1 - p1) ** (n)) * self.binom(
                n + self.nb_stage - 1, self.nb_stage - 1)
        return prob

# For Simulation of Selfish mining for nb_simulations
    def Simulate(self):
        while (self.__counter <= self.__nb_simulations):
            s = np.random.uniform(0, 1)  # random number for each simulation
            if self.currentstate==0:
               # s = np.random.uniform(0, 1)  # random number for each simulation
                prob=self.__prob_state0(self.__alpha) # probability Mb+Mc
                if s <= prob:
                    print("mined by selfish *************************************************************")
                # selfish pool finds a block and machine moves to state i==1
                # once the selfish has solved k stages, how many is solved by the honest? #Rn=k−stageh−1
                ### compute Mc where the honest pool has exactly solved k-1 stages
                    Mc = (self.__alpha ** self.R) * ((1 - self.__alpha) ** (self.nb_stage - 1)) * self.binom(
                        self.nb_stage + self.R - 2, self.R - 1)
                    s = np.random.uniform(0, )
                    if s<Mc:
                        self.__selfishValidBlocks+=1
                        self.R= self.nb_stage #honest pool has solved k-1 stages and only one is remained,
                        # selfish pool publishes his block and the remaining number of stages remain unchanged
                        self.previousstate = self.currentstate
                    else:
                     self.__privateChain += 1
                     self.stage_hon= self.__expectedstages(self.__alpha,self.R, self.nb_stage-1)
                     #honest pool has completedat most k−2 stages in the meantime
                     self.R=self.nb_stage-self.stage_hon -1 # Find the remaining number of stage_hon where is at maximum k-2
                     self.previousstate=self.currentstate
                     self.currentstate+=1

                else: # honest pool has been faster than the selfish pool. The state machine will loop in state
                     self.__honestsValidBlocks += 1
                     self.R = self.nb_stage
                     self.previousstate = self.currentstate

            elif self.currentstate -self.previousstate==1: #The machine has just moved forward to currentState>1.
                # The selfish andthe honest pool have to complete k and R stages to trigger the next state transition, respectively
                prob = self.__prob_moveforward(self.__alpha)  # probability Mb+Mc
                if s <= prob: #The selfish pool has been faster than the honest pool.
                    # The state machine will transition to state currentState+ 1.
                    self.__privateChain+=1
                    self.stage_hon=self.__expectedstages(self.__alpha,self.nb_stage,self.R) #honest pool has completed at most R−1 stages in the meantime
                    if self.currentstate==1: #honest pool needs to find R−stage_hon+ 1 more stages to
                        # complete its current Pow and trigger a state transition
                        self.R=self.R -self.stage_hon +1
                    else: # currentstate>=2
                        self.R=self.R -self.stage_hon # The honest pool needs to find R−stage_hon more stages
                        # to complete its current Pow and trigger a state transition
                    self.previousstate = self.currentstate
                    self.currentstate += 1

                else: #honest pool has been faster than the selfish pool.  Machine moves to currentState−1
                    self.__selfishValidBlocks += 1
                    self.__privateChain -= 1
                    self.stage_sel=self.__expectedstages(1-self.__alpha,self.R, self.nb_stage) #The selfish pool has completedat
                    # most k−1 stages in the meantime
                    self.R=self.nb_stage-self.stage_sel
                    self.previousstate = self.currentstate
                    self.currentstate -= 1

            else: # self.currentstate -self.previousstate==-1
                # machine has just moved back to currentState>1. Selfish pool has to complete R stages to trigger a state transition.
                # If currentStage= 1 (resp.currentStage>2), then the honest pool has to complete k−1 (resp.k) stages
                # to trigger the next state transition
                prob = self.__prob_moveback(1-self.__alpha)  # probability Mb+Mc
                if s <= prob:
                    self.publicchain+=1
                    if self.currentstate==1:
                        self.stage_hon=self.__expectedstages(self.__alpha,self.R, self.nb_stage-1)
                        # honest pool has completed at most k−2 stages in the meantime
                    else: #currentstate>=2
                        self.stage_hon=self.__expectedstages(self.__alpha,self.R, self.nb_stage)
                        #honest pool has completed at most k−1stages in the meantime
                    self.R=self.nb_stage-self.stage_hon #honest pool needs to find k−stageho nmore stages
                    # to complete its current PoW and trigger a state transition
                    self.previousstate = self.currentstate
                    self.currentstate += 1
                else: #honest pool has been faster than the selfish pool.
                # machine moves to currentState−1
                    self.__selfishValidBlocks += 1
                    self.__privateChain -= 1
                    if self.currentstate == 1:
                        # selfish pool hascompleted at mostR−1stages in the meantime
                        self.stage_sel=self.__expectedstages(1-self.__alpha,self.nb_stage-1, self.R-1)
                    else:  # currentstate>=2
                        #The selfish pool hascompletedat most R−1stages in the meantime
                        self.stage_sel=self.__expectedstages(1-self.__alpha,self.nb_stage, self.R-1)

                    self.R = self.nb_stage - self.stage_sel  # honest pool needs to find k−stageho nmore stages
                    # to complete its current PoW and trigger a state transition
                    self.previousstate = self.currentstate
                    self.currentstate -= 1
            self.__counter += 1
        while(self.__privateChain>0): # Publishing private chain if not empty when total nb of simulations reached
            self.__selfishValidBlocks += 1
            self.__privateChain-= 1
        self.actualize_results()
        if self.__revenue==None:
            self.__revenue=0
        return (self.__revenue, self.__probability())


# For rewards
    def actualize_results(self):
        # Total Blocks Mined
        self.__totalMinedBlocks = self.__honestsValidBlocks + self.__selfishValidBlocks
        # Orphan Blocks
        self.__orphanBlocks = self.__nb_simulations - self.__totalMinedBlocks
        print("number of orphan", self.__orphanBlocks, " --- number of stages", self.nb_stage)
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
    stage_No=[2,7,10,15,25]
    block_No = 1000
    count = 0  #pourcentage done
    container_block = []
    honestprob=[]
    plot = []

for stg in stage_No:
 print("stage---------------------------------------------------------------------: ", stg)
 for alpha in alphas:
    print("alpha------------------------------------------------------------------: ", alpha)
    new = Selfish_BlockMining(**{'nb_simulations': block_No, 'nb_stage': stg, 'alpha': alpha})
    container_block.append(new.Simulate()[0]) # append revenue
    honestprob.append(new.Simulate()[1])
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


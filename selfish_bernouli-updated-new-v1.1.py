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
        self.__counter = 1 # counts the number of simulations
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
        self.same=False #indicates that if two same event(like moving forward, moving back are happened in order)

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
        #once the selfish has solved k stages, how many is solved by the honest? honest may have R stages
        # from the last transition
        #if self.__privateChain==1 honest needs to solve R-1 stages and for higher states we consider R stages
        self.numerator=0
        self.division=0
        print("PREVIOUS r", self.R)

        for n in range(self.R):
            self.numerator = self.numerator + (n*(p1 ** self.nb_stage) * ((1 - p1) ** (n)) * self.binom(
                n + self.nb_stage - 1, self.nb_stage - 1))

        for m in range(self.R):
            self.division = self.division + ((p1 ** self.nb_stage) * ((1 - p1) ** (m)) * self.binom(
                m + self.nb_stage - 1, self.nb_stage - 1))

        if self.same:
            self.R= self.R-int(self.numerator/self.division)
        else: self.R= self.nb_stage-int(self.numerator/self.division)  # R=k-stageh

        if self.__privateChain==1: self.R=self.R-1
        return self.R

    def __residue_two(self,p1):
        ### moved back
        # honest pool solves R=k−stageh stages while selfish solves k stages for higher states
        self.numerator=0
        self.division=0

        for n in range(self.nb_stage):
            self.numerator = self.numerator + (n* (p1 ** self.R) * ((1-p1) ** (n)) * self.binom(
                n + self.R - 1, self.R - 1))
        for m in range(self.nb_stage):
             self.division = self.division + ((p1 ** self.R) * ((1-p1) ** (m)) * self.binom(
                 m + self.R - 1, self.R - 1))

        if self.same:
            self.R = self.R - int(self.numerator / self.division)# the machin moved back in the last transition
        # so there selfish may has already solved some stages, now is also moving back
        else:
            self.R = self.nb_stage - int(self.numerator / self.division)
        return self.R

    def __residue_three(self,p1):
        ### in state 0 , with probability of Mc honest pool has solved k-1 stages
        #residue_three finds the expected number of stages where at maximum k-2 stages solved by honest pool
        #print("remaing3----------------------------------------------------------------------", self.R)
        self.numerator=0
        self.division=0
        for n in range(self.nb_stage-1):
            self.numerator = self.numerator + (n* (p1 ** self.R) * ((1 - p1) ** (n)) * self.binom(
                n + self.R - 1, self.R - 1))
        for m in range(self.nb_stage-1):
            self.division = self.division + (p1 ** self.R) * ((1 - p1) ** (m)) * self.binom(
                m + self.R - 1, self.R - 1)
        self.R= self.nb_stage-int(self.numerator/self.division)-1
        return self.R

# For finding mining probability of pools
    def __prob_one(self, p1):
        ### machine moved forward, selfish pool has to solve k stages where honest solves R stages
        self.selfishprob = 0
        for n in range(self.R):
            self.selfishprob = self.selfishprob + (p1 ** self.nb_stage) * ((1 - p1) ** (n)) * self.binom(
                n + self.nb_stage - 1, self.nb_stage - 1)
        return self.selfishprob

    def __prob_two(self, p1):
        ### machine moved back, selfish solves his remaining number of stages untill honest solve his stages
        # where state i>1 selfish needs to finish his remaining stages where honest
        self.selfishprob = 0
        for n in range(self.nb_stage - 1):
         self.selfishprob = self.selfishprob + (p1 ** self.R) * ((1 - p1) ** (n)) * self.binom(
                n + self.R - 1, self.R - 1)
        return self.selfishprob

    def __prob_three(self, p1):
        ### in state 0
        self.selfishprob=0
        ### compute Mb+Mc
        for n in range(self.nb_stage):
            self.selfishprob = self.selfishprob + (p1 ** self.R) * ((1 - p1) ** (n)) * self.binom(
                    n + self.R - 1, self.R - 1)
        return self.selfishprob

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
        #machine has moved back
            prob = self.__probability(2)[0]
            #print("prob_back",prob)
        #machine has moved forward
        elif self.movedforward:
            prob = self.__probability(1)[0]
            #print("prob_forward",prob)
        else:
        #machine has started or looped in state 0
            prob = self.__probability(3)[0]
            #print("prob_loop",prob)
        return prob

# For Simulation of Selfish mining for nb_simulations
    def Simulate(self):
        while (self.__counter <= self.__nb_simulations):
            s = np.random.uniform(0, 1)  # random number for each simulation
            if s <= self.__simprobability():
                print("mined by selfish *************************************************************")
                if self.__privateChain==0:  #the machine just has started or looped in state 0
                    print("in state0----------------------------------------------------------")
                    # selfish pool finds a block and machine moves to state i==1
                    # once the selfish has solved k stages, how many is solved by the honest? #Rn=k−stageh−1
                    ### compute Mc where the honest pool has exactly solved k-1 stages
                    Mc = (self.__alpha ** self.R) * ((1 - self.__alpha) ** (self.nb_stage - 1)) * self.binom(
                        self.nb_stage + self.R - 2, self.R - 1)
                    s = np.random.uniform(0, self.__simprobability())
                    if s<Mc:
                        self.__selfishValidBlocks+=1
                        print('previousR', self.R)
                        self.R= self.nb_stage #honest pool has solved k-1 stages and only one is remained,
                        # selfish pool publishes his block and the remaining number of stages remain unchanged
                        print("remaining",self.R)
                    else:
                     # Find the remaining number of stages where is at maximum k-2
                     self.__privateChain += 1
                     print('previousR', self.R)
                     X=self.__residue(3,self.__alpha)
                     print("remaining",X)

                else: #self.privatechain>=1 machine is in state i>=1 and moves to state i>1
                    print("in state >=1---------------------------------------------------")
                    #the selfish pool has moved to the next state
                    print('previousR', self.R)
                    Z=self.__residue(1, self.__alpha)
                    print("remaining",Z)
                    if self.movedforward == True: self.same = True
                    self.__privateChain += 1  # appending the block into privatechain
                        # compute remaining number of stages for the next transation
                self.movedforward=True

            else:
                print("mined by honest **************************************************************")

                if self.__privateChain==0:
                        print("in state 0-----------------------------------------------------")
                        self.__honestsValidBlocks += 1
                        print('previousR', self.R)
                        self.R = self.nb_stage
                        print("remaining",self.R)

                if self.__privateChain==1:
                       print("in state=1------------------------------------------------------")
                       self.__selfishValidBlocks += 1
                       #self.__honestsValidBlocks +=1
                       self.publicchain = self.__privateChain
                       self.__privateChain-= 1
                       print('previousR', self.R)
                       M=self.__residue(2,1-self.__alpha)
                       print("remaining",M)

                       if self.movedback==True:
                           self.same=True #to consider
                           #self.movedback=False #to show its in state 0
                       # into account the solved stages before machine again move back
                       else: self.movedback=True

               ########### for cases that the number of unpublished block are greater equal >=2
                if self.__privateChain>1:
                   print("in state>1----------------------------------------------")
                   self.publicchain+=1
                   self.__selfishValidBlocks += 1
                   self.__privateChain-= 1
                   print('previousR', self.R)
                   T=self.__residue(2, 1-self.__alpha)
                   print("remaining", T)
                   if self.movedback == True:
                       print("true")
                       self.same = True  # to consider
                   # into account the solved stages before machine again move back
                   else:
                       self.movedback = True

                self.movedforward = False
            self.__counter += 1
        while(self.__privateChain>0): # Publishing private chain if not empty when total nb of simulations reached
            self.__selfishValidBlocks += 1
            self.__privateChain-= 1
        self.actualize_results()
        if self.__revenue==None:
            self.__revenue=0
        return (self.__revenue, self.__probability(0)[1])

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
    stage_No=[2,7]
    block_No = 10
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


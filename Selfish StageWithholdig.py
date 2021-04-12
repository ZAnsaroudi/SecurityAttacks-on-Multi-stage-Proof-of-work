import sys
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter1d
from mpmath import *

# Selfish minning on a sequential-mining multi-stage Proof-of-Work blockchain
class Selfish_StageWithholding:

    def __init__(self, **kwargs):
        # Set Parameters

        '''Selfish pool's network hashing power share = prob. the selfish pool finds a stage
        earlier than the honest pool'''
        self.__alpha = kwargs['alpha']
        self.__k = kwargs['k']
        self.__transitions_no = kwargs['transitions_no']
        self.__withholding = kwargs['withholding'] # withholding percent

        # For results
        self.__unpublished_blocks_no = 0
        self.__honest_reward = 0  # Number of blocks in the longest chain mined by the honest pool
        self.__selfish_reward = 0 # Number of blocks in the longest chain mined by the selfish pool
        self.__selfish_stagewithholder_reward = 0  # share of Stage-Withholder from the honest reward
        self.__selfish_stagewithholder_relative_reward = 0

        # For finding probabilities
        self.__R = self.__k

        # For monitoring the state machine
        self.__transition_counter = 0  # counts the number of simulations
        self.__current_state = 0
        self.__previous_state = 0
        self.__stage_sel = 0  # Expected number of stages solved by the selfish pool when it looses a mining race
        self.__stage_hon = 0  # Expected number of stages solved by the honest pool when it looses a mining race

    def probabilistic_round(self, x):
        return int(x + random.uniform(0, 1))
        # int(math.floor(x + random.random()))

    def binom(self, n, k):
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


    # For finding the expected number of stages completed by the losing pool
    def __expectedstages(self, p_success, successes_no, max_failures):
        '''Computing E[N | N <= max_failures], where N is negative binomial dist. counting the number
        of stages found by the losing pool (i.e., failures) while the winning pool found
        successes_no stages (i.e., successes).

        p_success is the success probability, that is, the probability that a new success occurs
        earlier than a new failure.
        '''
        numerator = 0
        denominator = 0
        for n in range(max_failures + 1):
            numerator = numerator + (n * (p_success ** successes_no) * ((1 - p_success) ** (n)) * self.binom(
                n + successes_no - 1, successes_no - 1))

        for n in range(max_failures + 1):
            denominator = denominator + ((p_success ** successes_no) * ((1 - p_success) ** (n)) * self.binom(
                n + successes_no - 1, successes_no - 1))
        x=numerator / denominator
        self.expected = int(self.probabilistic_round(x)) #round to the nearest integer
        return self.expected

    # For computing M_a, M_b, and M_c
    def __expression12(self, p1):  # Expression (12)
        '''The machine has just moved forward to a currentState >=1.
        The selfish and the honest pool have to complete k and R stages to trigger
        the next state transition, respectively.
        '''
        Ma = 0  # M_a
        for n in range(self.__R):
            Ma = Ma + (p1 ** self.__k) * ((1 - p1) ** (n)) * self.binom(
                n + self.__k - 1, self.__k - 1)

        return Ma

    def __expression13(self, p1):  # Expression (13)
        '''The machine has just moved back to a currentState >=1.
        The selfish and the honest pool have to complete R and R stages to trigger
        the next state transition, respectively.

        If currentStage = 1 (resp. currentStage > 2), then the honest pool has to
        complete k-1 (resp. k) stages to trigger the next state transition.
        '''

        Ma = 0  # M_a

        if self.__current_state == 1:
            No = self.__k - 1
        else:
            No = self.__k

        for n in range(No):
            Ma = Ma + (p1 ** self.__R) * ((1 - p1) ** (n)) * self.binom(
                n + self.__R - 1, self.__R - 1)

        return Ma

    def __expressions10and11(self, p1):  # Expressions (10) and (11)
        '''The machine has either just started, looped or returned to state 0 from state $1$.
        The selfish and the honest pool have to complete R and k stages to
        trigger the next state transition, respectively.
        '''
        Ma = 0  # M_a
        Mb = 0  # M_b
        Mc = 0  # M_c

        ''' Compute Ma as the probability that the selfish pool finds R stages and, 
        in the meantime, the honest pool finds k-2 stages at most.
        '''
        for n in range(self.__k - 1):
            Ma = Ma + (p1 ** self.__R) * ((1 - p1) ** (n)) * self.binom(
                n + self.__R - 1, self.__R - 1)

        ''' Compute Mb as the probability that the selfish pool finds R stages and, 
        in the meantime, the honest pool finds k-1 stages.
        '''
        Mb = (p1 ** self.__R) * ((1 - p1) ** (self.__k - 1)) * self.binom(
            self.__k + self.__R - 2, self.__R - 1)

        return Ma, Mb

        # Compute the selfish pool's expected relative reward when it does not attack and mines honestly.

    def __probability(self):
        p1 = (1-self.__withholding) * self.__alpha   # probability that the selfish pool finds a stage earlier than the honest pool

        # Selfish pool's expected relative reward = Selfish pool's mining probability
        selfishPoolMiningProb = 0
        for n in range(self.__k):
            selfishPoolMiningProb = selfishPoolMiningProb + (p1 ** self.__k) * ((1 - p1) ** (n)) * self.binom(
                n + self.__k - 1, self.__k - 1)
        return selfishPoolMiningProb

    # For Simulation of Selfish mining for transitions_no
    def Simulate(self):
        while (self.__transition_counter < self.__transitions_no):
            s = np.random.uniform(0, 1)  # new fresh random number for each iteration. 0 <= s < 1

            if self.__current_state == 0:
                '''The machine has either just started, looped or returned to state 0 from state $1$. 
                The selfish and the honest pool have to complete R and k stages to 
                trigger the next state transition, respectively.
                '''

                # Probability that the honest pool is faster than the selfish pool and mines the block
                Ma, Mb = self.__expressions10and11(self.__alpha)

                # Probability that the honest pool is faster than the selfish pool and mines the block
                Mc = 1 - Ma - Mb

                ''' 
                If 0 <= s < Ma, then, with probability Ma, the selfish pool 
                has found R stages and has completed the PoW, while the honest pool has found at most
                k-2 stages in the meantime.

                else if Ma <= s < Ma + Mb, then, with probability Mb, the selfish pool 
                has found R stages and has completed the PoW, while the honest pool has found k-1 stages
                in the meantime.

                else, with probability Mc = 1-Ma-Mb, the honest pool 
                has found k stages and has completed the PoW earlier than the selfish pool finds R stages.
                '''

                if s < Ma:  # With prob Ma:

                    self.__unpublished_blocks_no += 1  # The mined block is kept private

                    # In the meantime, the honest pool has found k-2 stages at most
                    self.__stage_hon = self.__expectedstages(self.__alpha, self.__R, self.__k - 2)

                    '''The loser honest pool has to find k-self.__stage_hon- 1 more stages to 
                    complete the second-last hash-puzzle in its current PoW and trigger a state
                    transition
                    '''
                    self.__R = self.__k - self.__stage_hon - 1

                    self.__previous_state = self.__current_state
                    self.__current_state += 1  # The state machine will move forward to state 1

                elif s < Ma + Mb:  # With prob Mb:

                    self.__selfish_reward += 1  # The selfish pool published the mined block

                    self.__previous_state = self.__current_state
                    # currentState is unchanged as the state machine will loop on state 0

                    self.__R = self.__k  # The selfish pool needs to find k more stages to trigger a new state transition


                else:  # With prob Mc:
                    self.__honest_reward += 1

                    self.__previous_state = self.__current_state
                    # currentState is unchanged as the state machine will loop on state 0.

                    self.__R = self.__k  # The selfish pool discards to find k more stages to trigger a new state transition


            elif self.__current_state - self.__previous_state == 1:  # The machine has just moved forward to currentState (such that currentState>=1).
                # The selfish and the honest pool have to complete k and R stages to trigger the next state transition, respectively

                Ma = self.__expression12(
                    self.__alpha)  # M_a = probability the selfish pool is faster than the honest pool

                ''' 
                If 0 <= s < Ma, then, with probability Ma, the selfish pool 
                has found k stages and has completed a PoW, while the honest pool has found at most R-1 stages 
                in the meantime.


                else, with probability Mb = 1-Ma the honest pool has found R stages and has completed a PoW, 
                while the honest pool has found at most k-1 stages in the meantime.
                '''

                if s < Ma:  # The selfish pool has been faster than the honest pool.

                    # The state machine will transition to state currentState+ 1.
                    self.__unpublished_blocks_no += 1  # The mined block is kept private

                    # In the meantime, the honest pool has found R-1 stages at most
                    self.__stage_hon = self.__expectedstages(self.__alpha, self.__k, self.__R - 1)

                    if self.__current_state == 1:  # The honest pool needs to find R−stage_hon+ 1 more stages to
                        # complete its current Pow and trigger a state transition
                        self.__R = self.__R - self.__stage_hon + 1

                    else:  # current_state>=2
                        self.__R = self.__R - self.__stage_hon  # The honest pool needs to find R−stage_hon more stages
                        # to complete its current Pow and trigger a state transition

                    self.__previous_state = self.__current_state
                    self.__current_state += 1

                else:  # honest pool has been faster than the selfish pool. The state machine will move back to currentState−1

                    self.__selfish_reward += 1  # The selfish pool releases the first unpublished block

                    self.__unpublished_blocks_no -= 1

                    self.__stage_sel = self.__expectedstages(1 - self.__alpha, self.__R,
                                                             self.__k - 1)  # The selfish pool has completed at
                    # most k−1 stages in the meantime

                    self.__R = self.__k - self.__stage_sel  # The selfish pool needs to find k - stage_sel more stages to
                    # complete its current PoW and trigger a state transition

                    self.__previous_state = self.__current_state
                    self.__current_state -= 1

            else:  # self.__current_state -self.__previous_state==-1

                # The machine has just moved back to currentState (such that currentState>=1). The selfish pool has to complete
                # R stages to trigger a state transition.
                # If currentState=1 (resp.currentState>2), then the honest pool has to complete k−1 (resp. k) stages
                # to trigger the next state transition

                Ma = self.__expression13(
                    self.__alpha)  # M_a = probability the selfish pool is faster than the honest pool

                honest_stages_to_comp = self.__k - 1 if self.__current_state == 1 else self.__k  # honest_stages_to_comp is only used for logging

                ''' 
                If 0 <= s < Ma, then, with probability Ma, the selfish pool 
                has found R stages and has completed a PoW. In the meantime, if currentState=1 (resp.currentState>2), 
                the honest pool has completed at most k−2 (resp. k-1) stages.


                else, with probability Mb = 1-Ma the opposite happened.
                '''

                if s < Ma:  # with probability Ma the selfish pool has been faster than the honest pool
                    # The machine will move forward to state currentState+1

                    self.__unpublished_blocks_no += 1  # The block is kept private
                    if self.__current_state == 1:
                        # the honest pool has completed at most k−2 stages in the meantime
                        self.__stage_hon = self.__expectedstages(self.__alpha, self.__R, self.__k - 2)

                    else:  # current_state>=2
                        # honest pool has completed at most k−1 stages in the meantime
                        self.__stage_hon = self.__expectedstages(self.__alpha, self.__R, self.__k - 1)

                    self.__R = self.__k - self.__stage_hon  # honest pool needs to find k−stage_hon more stages
                    # to complete its current PoW and trigger a state transition

                    self.__previous_state = self.__current_state
                    self.__current_state += 1

                else:  # the honest pool has been faster than the selfish pool.
                    # The machine will move back to currentState−1

                    self.__selfish_reward += 1  # The selfish pool releases the first unpublished block

                    self.__unpublished_blocks_no -= 1

                    if self.__current_state == 1:
                        # the selfish pool has completed at most R−1 stages in the meantime
                        self.__stage_sel = self.__expectedstages(1 - self.__alpha, self.__k - 1, self.__R - 1)
                    else:  # current_state>=2
                        # The selfish pool has completedat most R−1 stages in the meantime
                        self.__stage_sel = self.__expectedstages(1 - self.__alpha, self.__k, self.__R - 1)

                    self.__R = self.__k - self.__stage_sel  # the selfish pool needs to find k−stage_sel nmore stages
                    # to complete its current PoW and trigger a state transition

                    self.__previous_state = self.__current_state
                    self.__current_state -= 1

            # endif
            self.__transition_counter += 1

        # endwhile

        # Publishing the unpublishedBlockNo blocks
        self.__selfish_reward += self.__unpublished_blocks_no
        # Relative reward
        self.__selfish_stagewithholder_reward += self.__selfish_reward
        self.__selfish_stagewithholder_reward += (self.__withholding*self.__alpha/(self.__withholding*self.__alpha+(1-self.__alpha)))*self.__honest_reward
        self.__selfish_stagewithholder_relative_reward = self.__selfish_stagewithholder_reward /\
                                                         (self.__selfish_reward + self.__honest_reward)
        return (self.__selfish_stagewithholder_relative_reward, self.__probability()) #self.probability is the same as
        # the reward for honestly mining by the Selfish Stage-Withholder


###########################################################
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def setConfigParameters():

    '''Evaluate the attack profitability with respect to selfish pool's network hashing power share of values. 
    E.gThen, with scale_factor 100, we have alpha = 0.01, 0.02, 0.03, [...], 0.50Then, with scale_factor 1000 we 
    have alpha = 0.001, 0.002, 0.003, [...], 0.500.
    '''
    scale_factor = 1000
    alphas = np.linspace(1 / scale_factor, 0.5, int(scale_factor / 2))
    alphas.round(int(round(math.log(scale_factor, 10))), alphas)

    stage_no=15
    withholding_percents = [0, 0.1,0.3,0.5,0.7, 0.9]
    transitions_no = 100000
    stage_withholding_rel_reward_array = []  # Array that stores the selfish mining relative reward for all values in alphas and stage_no
    sel_pool_mining_prob_array = []  # Array that stores the selfish pool mining prob for all values in alphas and stage_no
    plot = []
    return alphas, stage_no, withholding_percents, transitions_no, stage_withholding_rel_reward_array, sel_pool_mining_prob_array, plot


def run_state_machine(alphas, stage_no, withholding_percents, transitions_no, stage_withholding_rel_reward_array,
                      sel_pool_mining_prob_array, plot):

    for wth in withholding_percents:
       print (wth)
       for alpha in alphas:
         # print("alpha ------------------------------------------------------------------: ", alpha)
         new = Selfish_StageWithholding(**{'transitions_no': transitions_no, 'k': stage_no, 'alpha': alpha, 'withholding': wth})
         selfish_stagewithholder_relative_reward, selfish_mining_prob = new.Simulate()
         stage_withholding_rel_reward_array.append(selfish_stagewithholder_relative_reward)
         sel_pool_mining_prob_array.append(selfish_mining_prob)


    ''' For each element k in the array stage_no, create in y_elem a distinct row in which store 
    only the selfish mining relative reward values for that k
    '''
    y_elem = list(divide_chunks(stage_withholding_rel_reward_array, len(alphas)))

    ''' For each element k in the array stage_no, create in y_prob a distinct row in which store 
    only the selfish pool's mining probabilities for that k
    '''
    y_prob = list(divide_chunks(sel_pool_mining_prob_array, len(alphas)))
    plt.plot(alphas, y_prob[0], ':k', label='Honest Mining')
    plt.legend()

    for i in range(len(y_elem)):
        if i==0:
            label='Selfish Mining'
        else:
            label='Selfish Stage-Withholding, τ=' + str(withholding_percents[i])
        plt.plot(alphas, y_elem[i], label=label)
        plt.legend()

    plt.xlabel('Pool size')
    plt.ylabel('Pool revenue')
    plt.title('Selfish Stage-Withholding, '+str(stage_no)+' stages')
    plt.savefig("stage_withholding_profitability.png")
    plt.show()

# For testing
configParams = setConfigParameters()
run_state_machine(configParams[0], configParams[1], configParams[2], configParams[3], configParams[4], configParams[5],
                  configParams[6])

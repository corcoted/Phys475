
# coding: utf-8

# # Quantum Guessing Game
# by Ted Corcovilos 20160610
# 
# The computer picks a random quantum spin-1/2 state from a predetermined list.
# The player requests measurements (Pauli operators Sx, Sy, or Sz) and attempts to guess the state.

import random #for generating random numbers for the measurements
random.seed() # set a seed based on current time

import numpy as np

import matplotlib.pyplot as plt

# ## Game rules
# The cell below contains the rules for the game: List of possible states, penalty for incorrect guess, and the type of die to use.

#States are stored as an array of Bloch vectors (length of vectors must be <= 1)
#states = np.array([[ 0.68106963, -0.06972718,  0.72889113],
# [ 0.69532885, -0.42901413, -0.57659749],
# [-0.97751342,  0.00342076, -0.21084547],
# [-0.51890684,  0.77233682, -0.36637621],
# [-0.63303414, -0.70602328, -0.31748843],
# [-0.38039451, -0.905975  , -0.18576685],
# [-0.62914552,  0.591202  , -0.50463463],
#[ 0.41936635,  0.23636217, -0.87650715]])
#states = np.array([[1.,1.,1.],
#                   [1.,-1.,-1.],
#                   [-1.,1.,-1.],
#                   [-1.,-1.,1.]])/np.sqrt(3.) #vertices of tetrahedron
states =np.array(
[[-0.41120177,  0.08130806,  0.90791085],
 [-0.45270333, -0.5621289 ,  0.6921494 ],
 [ 0.39064291,  0.05635696, -0.91881555],
 [-0.36688601,  0.33389421, -0.86827951],
 [ 0.65944473,  0.0150693 ,  0.751602  ],
 [ 0.94049809, -0.33945593, -0.01526501],
 [-0.60580469,  0.42534142,  0.67237293],
 [-0.58226931, -0.66697171, -0.46487761]])
penalty = 5 #points for incorrect guess
maxguess = 200 # maximum number of measurements allowed
dsize = 20 # size of the dice used for rolling

Nstates = states.shape[0]

measurements = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]) #measurement axes (Bloch sphere rep. of Pauli matrices)
Nmeasurements = measurements.shape[0] # number of possible measurements

#Build lookup table (probablility of measuring + for each state) [axis, states]
lookup=np.zeros((measurements.shape[0],Nstates),dtype=int)
for i in range(measurements.shape[0]):
    for j in range(Nstates):
        lookup[i, j] = np.round(0.5*(1.0+np.dot(states[j],measurements[i]))*dsize)


#print(lookup) #print the lookup table.  Row is a measurement direction (x, y, z). Column is a state.  If the die roll is <= this number, the result is +. 

def measure(state, axis):
    """
    Perform a quantum measurement on a state vector along an axis vector. Returns +/-1.
    """
    #TODO simulate dice rolls
    if (random.randint(1,dsize)<=lookup[axis,state]):
        out = 1
    else:
        out = -1
    return out

def play(): #play the game!
    """
    Play the Game!
    """
    guesses = 0
    resultTable = np.zeros((measurements.shape[0],2),dtype=int)
    goal = random.randrange(Nstates)
    while (True): #main measurement loop
        print("Measurement number {0}".format(guesses))
        test = -1
        while test == -1: #make sure we have a valid input
            print("Which measurement do you want: x, y, z, (g)uess or (q)uit? ")
            key = input("Input: ")
            test = {
                'x': 0,
                'y': 1,
                'z': 2,
                'g': -10,
                'q': -99
            }.get(key,-1)
        if(test==-99):
            print("Quit.  State was {0}: {1}".format(goal,states[goal]))
            break
        if(test==-10): # Handle guess
            print("Possible states are")
            for i in range(0,Nstates,1):
                print("{0}: {1}".format(i,states[i]))
            key = int(input("Guess? "))
            if (key==goal):
                print("Correct! Final score: {}".format(guesses))
                break
            else:
                print("Wrong! Penalty: {}".format(penalty))
                guesses = guesses+5
                continue
        guesses = guesses+1
        result = measure(goal,test)
        resultTable[test,(1+result)//2] = resultTable[test,(1+result)//2]+1
        Ntests = resultTable[test,0]+resultTable[test,1]
        print("Measurement result: {0}".format(result))
        if (guesses>maxguess):
            print("Exceeded measurement limit!  Guesses = {}".format(guesses))
            break
        print("Result table: ")
        print(("{:>5} "*4).format("","x","y","z"))
        print(("{:>5} "*4).format("+",resultTable[0,1], resultTable[1,1], resultTable[2,1]))        
        print(("{:>5} "*4).format("-",resultTable[0,0], resultTable[1,0], resultTable[2,0]))
    return guesses, goal

def player(auto=False, aggressiveness=1.0):
    '''Take the role of Experimenter.'''
    #initialize some stuff
    #aggressiveness = 1.0 # increase this to make more risky guesses (default = 1)
    probs = lookup/(1.0*dsize) # convert dice table to probabilities
    count = 0
    labels =['X','Y','Z'] # some text strings for display
    if (auto): # pick a state
        secret = np.random.randint(0,Nstates) # pick a state
    resultTable = np.zeros((Nmeasurements,2),dtype=int)
    likelihood = np.ones(Nstates,dtype=float) #initialize the likelihood function
    while (count<=maxguess): #main loop
        #print(likelihood) # for debugging
        order = np.argsort(-1.0*likelihood) # order of likelihood values, greatest first
        #check other states
        guess = -1; measureq = -1 #initialize flags
        #if likelihood[order[0]]/likelihood[order[1]]>=(penalty-1)/aggressiveness: #check likelihood ratio to next most likely state and compare to risk
        if likelihood[order[0]]*aggressiveness>=(penalty-1)*(np.sum(likelihood)-likelihood[order[0]]): #check likelihood ratio to all states
            guess = order[0] # we're going to guess this state
        #temp fix: pick random measurement
        #measure = np.random.randint(0,3)
        com = (likelihood*states.T).sum(axis=1)
        mom2 = (likelihood*(states.T**2)).sum(axis=1)
        moments = mom2-com**2
        measureq = np.argmax(moments) # Measure along the direction with the largest moment of inertia
        #print("Center of mass: {}".format(com))
        #print("Moments: {}".format(moments))
        # Interact with user
        if guess==-1: #we're making a measurement
            if (auto): # autoplay
                test = measure(secret,measureq)
            else:
                print("The count is {}.  I'd like to measure {}, please.".format(count,labels[measureq]))
                print("What is the result? [p]lus or [m]inus or [q]uit?")
                test = 0
                while test == 0: #make sure we have a valid input
                    key = input("Input: ")
                    test = {
                        'p': 1,
                        'm': -1,
                        'f': 2,
                        'q': -99
                    }.get(key,0)
                if test == -99:
                    print("Quitting.  Count was {}.".format(count))
                    break
                if test == 2:
                    print("Forcing a guess.  Here we go!")
                    guess = order[0]
            count = count + 1
            if test == 1: # the measurement was +
                likelihood = likelihood*probs[measureq,:]
            if test == -1: # the measurement was -
                likelihood = likelihood*(1.0-probs[measureq,:])
        if guess>-1: # we're making a guess
            if (auto): # autoplay
                if guess==secret:
                    break
                else:
                    count = count + penalty
                    likelihood[guess] = 0
                    continue
            else:
                print("The list of states is")
                for i in range(0,Nstates,1):
                    print("{0}: {1}".format(i,1+states[i]))
                print("I'd like to guess state {}.  Is this correct? [y/n] or [q] for quit".format(guess))
                test = -1
                while test == -1: #make sure we have a valid input
                    key = input("Input: ")
                    test = {
                        'y': 0,
                        'n': 1,
                        'q': -99
                    }.get(key,-1)
                if test == -99:
                    print("Quitting.  Count = {}.".format(count))
                    break
                if test == 0:
                    print("Yay! Final Count = {}".format(count))
                    break
                if test == 1:
                    print("Oh, no!")
                    count = count + penalty
                    likelihood[guess] = 0.0
                    if sum(likelihood) == 0:
                        print("Something isn't right.  None of the states are left.  I'm giving up.")
                        break
                    continue
    if count>maxguess:
        count = maxguess
    return count, guess

def xlogx(x):
    # calculate x log2 (x), catching possible zero arguments
    number = len(x)
    output = np.nan
    if number == 1: # scalar case
        if x == 0:
            output = 0
        else:
            output = x * np.log2(x)
    else: # assuming 1d array
        output = 0.0*x # initialize array the same size as x
        for i in range(number):
            if x[i] == 0:
                output[i] = 0
            else:
                output[i] = x[i] * np.log2(x[i])
    return output
    

def player2(auto=False, aggressiveness=1.0):
    '''Take the role of Experimenter.'''
    #initialize some stuff
    #aggressiveness = 1.0 # increase this to make more risky guesses (default = 1)
    probs = lookup/(1.0*dsize) # convert dice table to probabilities
    count = 0
    labels =['X','Y','Z'] # some text strings for display
    if (auto): # pick a state
        secret = np.random.randint(0,Nstates) # pick a state
    resultTable = np.zeros((Nmeasurements,2),dtype=int)
    likelihood = np.ones(Nstates,dtype=float)/Nstates #initialize the likelihood function (normalized)
    exp_entropies = 1e9*np.ones(Nmeasurements+Nstates) # initialize expectation value of entropy array
    while (count<=maxguess): #main loop
        #print(likelihood) # for debugging
        order = np.argsort(-1.0*likelihood) # order of likelihood values, greatest first
        #check other states
        guess = -1; measureq = -1 #initialize flags
        entropy = -np.sum(xlogx(likelihood)) # Shannon entropy 
        if np.isnan(entropy):
            entropy = 0 #(replace nan with zero)
        #print("Current entropy = {}".format(entropy))
        # calculate expectation value of entropy for each measurement
        for i in range(Nmeasurements):
            expect_plus = probs[i,:]*likelihood
            expect_minus = (1.0-probs[i,:])*likelihood
            N_plus = np.sum(expect_plus) # normalization of plus
            N_minus = 1.0 - N_plus
            l_afterplus = expect_plus/N_plus
            l_afterminus = expect_minus/N_minus
            exp_entropies[i] = -N_plus*np.sum(xlogx(l_afterplus)) - N_minus*np.sum(xlogx(l_afterminus))
        for i in range(Nstates): # expectation value of entropy for guesses
            index = Nmeasurements+i
            # entropy after correct guess is zero
            if likelihood[i] == 0: # this state has already been excluded
                exp_entropies[index] = entropy # no change
                continue
            en_after_good = 0.0
            l_after_bad = 1.0*likelihood
            l_after_bad[i] = 0.0 # replace bad guess with zero
            l_after_bad = l_after_bad/np.sum(l_after_bad) # renormalize
            en_after_bad = -np.sum(xlogx(l_after_bad))
            exp_entropies[index] = likelihood[i]*en_after_good+(1.0-likelihood[i])*en_after_bad # todo weight for penalty
        delta_entropy = exp_entropies-entropy
        delta_entropy[Nmeasurements:] = aggressiveness/penalty*delta_entropy[Nmeasurements:]
        entropy_order = np.argsort(delta_entropy)
        # guess decision in next line
        if entropy_order[0]<Nmeasurements:
            measureq = entropy_order[0] # we're going to measure this direction
        else:
            guess = entropy_order[0]-Nmeasurements # we're going to guess this state
        # Interact with user
        if guess==-1: #we're making a measurement
            if (auto): # autoplay
                test = measure(secret,measureq)
            else:
                print("The count is {}.  I'd like to measure {}, please.".format(count,labels[measureq]))
                print("What is the result? [p]lus or [m]inus or [q]uit?")
                test = 0
                while test == 0: #make sure we have a valid input
                    key = input("Input: ")
                    test = {
                        'p': 1,
                        'm': -1,
                        'f': 2,
                        'q': -99
                    }.get(key,0)
                if test == -99:
                    print("Quitting.  Count was {}.".format(count))
                    break
                if test == 2:
                    print("Forcing a guess.  Here we go!")
                    guess = order[0]
            count = count + 1
            if test == 1: # the measurement was +
                likelihood = likelihood*probs[measureq,:]
                resultTable[measureq,0] = resultTable[measureq,0]+1
            if test == -1: # the measurement was -
                likelihood = likelihood*(1.0-probs[measureq,:])
                resultTable[measureq,1] = resultTable[measureq,1]+1
        if guess>-1: # we're making a guess
            if (auto): # autoplay
                if guess==secret:
                    break
                else:
                    count = count + penalty
                    likelihood[guess] = 0
                    continue
            else:
                print("The list of states is")
                for i in range(0,Nstates,1):
                    print("{0}: {1}".format(i,1+states[i]))
                print("I'd like to guess state {}.  Is this correct? [y/n] or [q] for quit".format(guess))
                test = -1
                while test == -1: #make sure we have a valid input
                    key = input("Input: ")
                    test = {
                        'y': 0,
                        'n': 1,
                        'q': -99
                    }.get(key,-1)
                if test == -99:
                    print("Quitting.  Count = {}.".format(count))
                    break
                if test == 0:
                    print("Yay! Final Count = {}".format(count))
                    break
                if test == 1:
                    print("Oh, no!")
                    count = count + penalty
                    likelihood[guess] = 0.0
                    if sum(likelihood) == 0:
                        print("Something isn't right.  None of the states are left.  I'm giving up.")
                        break
                    continue
        likelihood = likelihood/np.sum(likelihood) # renormalize the likelihood array
    if count>maxguess:
        count = maxguess
    return count, guess

def player3(auto=False, aggressiveness=1.0):
    '''Take the role of Experimenter.'''
    #initialize some stuff
    #aggressiveness = 1.0 # increase this to make more risky guesses (default = 1)
    quitit = False
    probs = lookup/(1.0*dsize) # convert dice table to probabilities
    count = 0
    labels =['X','Y','Z'] # some text strings for display
    if (auto): # pick a state
        secret = np.random.randint(0,Nstates) # pick a state
    resultTable = np.zeros((Nmeasurements,2),dtype=int)
    likelihood = np.ones(Nstates,dtype=float)/Nstates #initialize the likelihood function (normalized)
    exp_entropies = 1e9*np.ones(Nmeasurements) # initialize expectation value of entropy array
    while (count<=maxguess): #main loop
        #print(likelihood) # for debugging
        order = np.argsort(-1.0*likelihood) # order of likelihood values, greatest first
        #check other states
        guess = -1; measureq = -1 #initialize flags
        entropy = -np.sum(xlogx(likelihood)) # Shannon entropy 
        if np.isnan(entropy):
            entropy = 0 #(replace nan with zero)
        print("Current entropy = {}".format(entropy))
        # calculate expectation value of entropy for each measurement
        for i in range(Nmeasurements):
            expect_plus = probs[i,:]*likelihood
            expect_minus = (1.0-probs[i,:])*likelihood
            N_plus = np.sum(expect_plus) # normalization of plus
            N_minus = 1.0 - N_plus
            l_afterplus = expect_plus/N_plus
            l_afterminus = expect_minus/N_minus
            exp_entropies[i] = -N_plus*np.sum(xlogx(l_afterplus)) - N_minus*np.sum(xlogx(l_afterminus))
        #print("Exp entropy = {}".format(exp_entropies))
        delta_entropy = exp_entropies-entropy
        entropy_order = np.argsort(delta_entropy)
        # guess decision in next line
        if likelihood[order[0]]*aggressiveness < likelihood[order[1]]*penalty: # likelihood ratio test
            measureq = entropy_order[0] # we're going to measure this direction
        else:
            guess = order[0] # we're going to guess this state
        # Interact with user
        if guess==-1: #we're making a measurement
            if (auto): # autoplay
                test = measure(secret,measureq)
            else:
                print("The count is {}.  I'd like to measure {}, please.".format(count,labels[measureq]))
                print("What is the result? [p]lus or [m]inus or [q]uit?")
                test = 0
                while test == 0: #make sure we have a valid input
                    key = input("Input: ")
                    test = {
                        'p': 1,
                        'm': -1,
                        'f': 2,
                        'q': -99
                    }.get(key,0)
                if test == -99:
                    print("Quitting.  Count was {}.".format(count))
                    quitit = True
                    break
                if test == 2:
                    print("Forcing a guess.  Here we go!")
                    guess = order[0]
            count = count + 1
            if test == 1: # the measurement was +
                likelihood = likelihood*probs[measureq,:]
                resultTable[measureq,0] = resultTable[measureq,0]+1
            if test == -1: # the measurement was -
                likelihood = likelihood*(1.0-probs[measureq,:])
                resultTable[measureq,1] = resultTable[measureq,1]+1
        if guess>-1: # we're making a guess
            if (auto): # autoplay
                if guess==secret:
                    break
                else:
                    count = count + penalty
                    likelihood[guess] = 0
                    continue
            else:
                print("The list of states is")
                for i in range(0,Nstates,1):
                    print("{0}: {1}".format(i,states[i]))
                print("I'd like to guess state {}.  Is this correct? [y/n] or [q] for quit".format(guess))
                test = -1
                while test == -1: #make sure we have a valid input
                    key = input("Input: ")
                    test = {
                        'y': 0,
                        'n': 1,
                        'q': -99
                    }.get(key,-1)
                if test == -99:
                    print("Quitting.  Count = {}.".format(count))
                    quitit = True
                    break
                if test == 0:
                    print("Yay! Final Count = {}".format(count))
                    break
                if test == 1:
                    print("Oh, no!")
                    count = count + penalty
                    likelihood[guess] = 0.0
                    if sum(likelihood) == 0:
                        print("Something isn't right.  None of the states are left.  I'm giving up.")
                        break
                    continue
        likelihood = likelihood/np.sum(likelihood) # renormalize the likelihood array
    if count>maxguess:
        count = maxguess
    return count, guess, quitit


def testit(aggro=3.0,Nruns=1000):
    # run tests many times and calculate average score
    countlist = np.zeros((maxguess+1,Nstates),dtype=int)
    for x in range(Nruns):
        count, guess = player3(aggressiveness=aggro, auto=True)
        countlist[count,guess] += 1
        mean = sum(countlist.sum(axis=1)*np.arange(0,maxguess+1,1))/(x+1)
        stddev = np.sqrt(sum(countlist.sum(axis=1)*(np.arange(0,maxguess+1,1)-mean)**2)/(x+1))
    return mean, stddev, countlist


# Try changing `aggro` below to see the effect of more aggressive guessing.  Note that for `aggro<4.5` or so, there are games that never get solved.  Is this a coding error or real?

def testaggro():
    # aggro around 3.0 gave best results in quick scan
    # my code here
    npoints = 31
    alist = np.linspace(2.0,5.0,npoints)
    m = np.zeros(npoints)
    s = np.zeros(npoints)
    clist = np.zeros((maxguess+1,npoints))
    for i in range(npoints):
        mi, si, ci = testit(aggro = alist[i],Nruns=1000)
        print("Aggro = {}. Results: mean = {}, stddev = {}".format(alist[i],mi,si))
        m[i] = mi
        s[i] = si
        clist[:,i] = ci.sum(axis=1)
    #for i in range(npoints):
    #    plt.plot(clist[:51,i], label=alist[i])
    plt.plot(m)
    plt.plot(s)
    #plt.legend()
    plt.show()

def main():
    q = False
    total = 0
    while not q:
        c, g, q = player3(aggressiveness=4.0,auto=False)
        total = total + c
        print("Total score = {}".format(total))


if __name__ == "__main__":
    main()
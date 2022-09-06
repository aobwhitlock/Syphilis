import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#Initial population values. Each class begins with one case of primary syphilis
E = 0 #exposed
YP = 1 #primary syph
YS = 0 #secondary syph
L = 0 # late latent syph
EL = 0 #early laten syph
YRS = 0 #recurrent secondary syph
Z = 0 #partial immunity (temporary)
Yt = 0 #tertiary syph

#Activity class values
Ntot = 100000 #total population size

#Size of population for classes 1, 2, and 3
nClasses = 3
N1 = 0.72*Ntot     
N2 = 0.15*Ntot         
N3 = 0.13*Ntot 

#Number of partnerships for classes 1, 2, and 3
C1 = 2 
C2 = 8
C3 = 30
totC = (N1*C1)+(N2*C2)+(N3*C3) #total partnerships

#make a list of lists of parameters for each sexual behaviour class
class1 = [(N1*C1/totC), N1, C1]
class2 = [(N2*C2/totC), N2, C2]
class3 = [(N3*C3/totC), N3, C3]
sexclass = [class1, class2, class3] 


#initial compartment values for each class in a list
init1 = [N1-1, E, 1, YS, L, EL, YRS, Z, Yt] #values for class 1
init2 = [N2-1, E, 1, YS, L, EL, YRS, Z, Yt] #values for class 2
init3 = [N3-1, E, 1, YS, L, EL, YRS, Z, Yt] #values for class 3
init = init1 + init2 + init3 #make a single list containing all values
             
#progressions and rates
sig1 = 1/21 #Average incubation period, days, E -> YP
sig2 = 1/46 #Average time, days, spent YP -> YS
sig3 = 1/108 #Average time, days, spent YS -> L and YS -> EL and YRS -> L
sig4 = 0.5 # Rate per year EL -> YRS 
sig5 = 0.033 #Rate per year L -> Yt
gamma = 0.2 #Average rate of loss of immunity (ollowing treatment
nu = 0.25 #Proportion of individuals in 2° stage who progress to early latent stage	
beta = 0.003042 #calibrated value
m = 0.033 #Entry/exit rate
tau = 0 #Average treatment rate

dt = 0.002 #step size

#These are the state values at equilibrium, included here to save time and avoid the burn-in step. To do the burn in, just comment this out.
init = [71992.9265552007,
 2.8953911652140776,
 2.518778955310003,
 1.2957175588299656,
 0.14567462677503484,
 0.0056272911838524155,
 0.06658057053580357,
 0.0,
 0.14567462677336082,
 14994.107199437365,
 2.412115055899345,
 2.0983640185071133,
 1.0794464904776402,
 0.12135975433252022,
 0.00468802763218677,
 0.055467461028845325,
 0.0,
 0.12135975433115345,
 12980.869066215355,
 7.830913829218044,
 6.812323388585295,
 3.504415110505866,
 0.3939935519192813,
 0.015219647307794064,
 0.18007470521772495,
 0.0,
 0.39399355191465546]


def transmission(time, dt, current_state, sexclass, beta, m, tau, sig1, sig2, sig3, sig4, sig5, gamma, nu):
    '''
    Compartmental model using Euler
    Input:
    time: duration
    dt: step size
    current_state: flat list containing current population values for all classes
    sexclass: list of lists, with each list containing parameters for one behaviour class
    constant model parameters: beta:nu
    
    Output:
    current_state: flat list containing updated values of current_state
    inf: list of incidence at each time step 

    '''
    steps = int(time/dt)
    substeps = int(steps/time)
    slices = np.linspace(0, len(current_state), len(sexclass)+1) #get indices of current_state corresponding to each class. len(sexclass) is number of classes. +1 to give endpoint
    inf = []
    for i in range(time):
        inf.append(current_state[2]+current_state[11]+current_state[20])
        for _ in range(substeps): #iterate through the substeps at each time point
            lambdas = get_lambda(current_state, sexclass, beta) #Get lambda value for all activity classes
            nStep = [] #Initialize vector to be returned with the next set of values
            for i in range(len(sexclass)): # Do the step for each sex activity class
                X, E, YP, YS, L, EL, YRS, Z, Yt = current_state[int(slices[i]):int(slices[i+1])] #Unpack the values of current_state for each class
                lam = lambdas[i]
                N = sexclass[i][1]
                fX = X + (m*N + tau*YP + tau*YS + gamma*Z - (lam+m)*X)*dt
                fE =  E + (lam*X-(sig1+m)*E)*dt
                fYP = YP + (sig1*E -(m+tau+sig2)*YP)*dt
                fYS = YS + (sig2*YP - (m+tau+sig3)*YS)*dt
                fEL =  EL + (nu*sig3*YS - (m + tau + sig4)*EL)*dt
                fYRS = YRS + (sig4*EL - (m + tau + sig3)*YRS)*dt
                fL = L + (sig3*YRS + (1 - nu)*sig3*YS - (sig5 + m + tau)*L)*dt
                fYt = Yt + (sig5*L - (m + tau)*Yt)*dt
                fZ = Z + (tau*L + tau*Yt + tau*YRS + tau*EL - (gamma + m)*Z)*dt
                nStep += [fX, fE, fYP, fYS, fL, fEL, fYRS, fZ, fYt]
            current_state = nStep
    return current_state, inf

def get_lambda(current_state, sexclass, beta):    
    '''
    Called by transmission() to calculate the lambda for each activity class
    Input:
    current state: list of current state for each behaviour class
    sexclass: list of lists containing parameters for each class 
    beta: probability of transmission per partnership
    
    Output:
    list of force of infection for each class
    '''
    propInf = 0
    slices = np.linspace(0, len(current_state), len(sexclass)+1) #get indices of current_state corresponding to each class. len(sexclass) is number of classes. +1 to give endpoint
    cList = []
    for i in range(len(sexclass)): #for each class
        prop, N, C = sexclass[i] #unpack class parameters
        X, E, YP, YS, L, EL, YRS, Z, Yt = current_state[int(slices[i]):int(slices[i+1])] #unpack class state
        yInf = (YP+YS+YRS)/N #get class contribution to lambda
        classProp = prop*yInf
        cList.append(C) #Store the partner value in a list so the entire list doesn't have to be unpacked twice
        propInf += classProp  
    lamList = []
    for C in cList: #Get lambda for each class 
        lam = C*beta*propInf
        lamList.append(lam) 
    return lamList


##################################
## create equilibrium population##
##################################
#time = 200000
#print("Burning in population")
#eqVals, inf = transmission(time, dt, init, sexclass, beta, m, tau, sig1, sig2, sig3, sig4, sig5, gamma, nu)
#print("Equilibrium incidence per 100,000 = ", inf[-1])

print("Beginning experiment")
time = 10

tau = 1.51 #screened population
resultTr, infTr = transmission(time, dt, init, sexclass, beta, m, tau, sig1, sig2, sig3, sig4, sig5, gamma, nu)

tau = 0 #unscreened population
resultUn, infUn = transmission(time, dt, init, sexclass, beta, m, tau, sig1, sig2, sig3, sig4, sig5, gamma, nu)

print("Plotting effects of intervention")
Years = list(range(time)) #Make a list of years for X axis

#get relative rate
relInc = 100/infUn[-1]
relInf = [i*relInc for i in infTr] #transform incidence rate 

plt.figure(figsize=(10,5))
plt.subplot(1,2,2)
plt.plot(Years, relInf)
plt.grid(linewidth=0.1)
plt.ylabel('% of pre-intervention incidence')
plt.xlabel('Time in years')
plt.title('B. Relative incidence following intervention')


plt.subplot(1,2,1)
plt.plot(Years, infTr, label='Intervention')
plt.plot(Years, infUn, label='No intervention')
plt.grid(linewidth=0.1)
plt.legend()
plt.ylabel('Incidence per 100,000')
plt.xlabel('Time in years')
plt.title('A. Syphilis incidence')
plt.savefig('IncidenceReduction.png')
plt.clf()

print("Plotting screening interval that leads to eradication")
taus = np.linspace(1, 12, 100) #get 50 values for tau between 1 and 12 
time = 20
inf20 = [] #make an empty list which will have the incidence at time 20 for each tau

#test each value
for tau in taus:
    endstate, inf = transmission(time, dt, init, sexclass, beta, m, tau, sig1, sig2, sig3, sig4, sig5, gamma, nu)
    inf20.append(inf[-1])

tInt = [12/i for i in taus] #convert tau to screening intervals

# plot screening interval
plt.figure(figsize=(5,5))
plt.plot(tInt, inf20)
plt.grid(linewidth=0.1)
plt.ylabel('Incidence at 20 years')
plt.xlabel('Screening interval in months')
plt.axhline(y = 0.1, color = 'r', linestyle = ':')
plt.annotate('Eradication threshhold', xy=(1.15, 0.104), xytext=(1.15, 0.104))
plt.title('Effect of screening interval on syphillis eradication')
plt.savefig('ScreeningIntervalEffect.png')
plt.clf()

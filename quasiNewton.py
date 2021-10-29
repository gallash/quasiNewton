# Quasi-Newton 

'''
This quasi-Newton optimization algorithm was built with the bissection method to perform the updates to the Hessian.

Feel free to change the function to be investigated in 'function' function. By default, this script will minimize the equation: (x1−3)**4+(x1−3*x2)**2.
If a new equation is added, please follow read the instructions presented in the 'function' function definition.

Dependency on sympy library.
'''

import sympy
import numpy as np
import pandas as pd
from timeit import default_timer as time


# Helping functions
def sym_gradient(function,variables):
    return [sympy.diff(function,x) for x in variables]


# Symbolic functions to be minimized
def function(opt,variables):
    y = 0
    # Minimize (x1−3)**4 + (x1−3x2)**2
    if (opt == 0):
        y = (variables[0] - 3)**4 + (variables[0] - 3*variables[1])**2

    # Add if's like the above to add more functions to be minimized   
    # If another equation is added or if the default one is modified, please 
    # modify the 'var' and 'init' functions accordingly, as well as the 'quests' 
    # list-variable right above the 'for' loop, adding to it the number referring
    # to the new equation to be optimized. 
    return y


# Creating symbolic variables for the function to be minimized
def var(opt):
    if (opt == 0):
        x1 = sympy.Symbol('x1')
        x2 = sympy.Symbol('x2') 
        X = [x1,x2]

    # Add if's like the above to add the variables to the function to be optimized here
    return X


# Starting values, the algorithm will start there
def init(opt):
    if (opt == 0):
        value = [0,3]
        a = 0 # 'a' and 'b' can be changed to other ranges
        b = 3

    # Add if's like the above to add starting points to additional functions to be 
    # optmized here
    return value, a, b


# Minimization function
def biss(a,b,EPS,x,y,d,variables):
    l = sympy.Symbol("l") # symbolic Lambda
    cnt = n = 0
    dy = 1
    obj = EPS/(b-a)
    while ((1/2)**n >= obj):
        n += 1
    
    aux = {}
    for i in range(len(variables)):
        aux[variables[i]] = x[i] + l*d[i]
    
    grady = sympy.diff(y.subs(aux))
    while ((abs(a - b) > EPS)and(cnt <= n)):
        Lambda = (a + b)/2
        dy = grady.subs({l:Lambda})
        if (dy < 0): a = Lambda
        if (dy > 0): b = Lambda
        cnt += 1
    
    return Lambda


class Point:
    def __init__(self):
        self.y = []
        self.norm = []


quests = [0] # Add here the number referring to the new equation to be optimized
instance = {}
cols = ['y', 'norm']
start = time()
for opt in quests:
    print("Equation",opt,"...",end=" ")

    # Variables declaration
    EPS = 1E-3 # Tolerance
    variables = var(opt)
    x, a, b = init(opt)
    dictx = {} # Creating dict {var_simb:value}
    D = np.identity(len(variables))
    xx = x
    k = j = 0
    N = len(variables)

    y = function(opt,variables)
    grady = sym_gradient(y,variables) # Gradient declaration
    grady_value = None
    instance[opt] = Point()
    while (True): 
        for n in range(len(variables)): # Loading value to dictx
            dictx[variables[n]] = xx[n]
        
        if grady_value is None: grady_value = np.array( [grady[n].subs(dictx) for n in range(len(grady))] )

        norm = np.sqrt(float(np.sum(np.power(grady_value,2))))
        
        if (norm <= EPS): break
        else:
            d = np.dot(-D,grady_value)
            Lambda = biss(a,b,EPS,x,y,d,variables) 
            p = Lambda*d
            xx_previous = xx
            xx += p
            grady_previous = grady_value
            for n in range(len(variables)): #  Loading value to dictx
                dictx[variables[n]] = xx[n]
            
            grady_value = np.array( [grady[n].subs(dictx) for n in range(len(grady))] )
            if(j == N):
                # j = 1
                j = 0
                x = xx
                k += 1
                continue        
            if (j < N):
                q = grady_value - grady_previous
                j += 1

                p_ = p.reshape(len(p),1)
                p_T = p.reshape(1,len(p))
                q_ = q.reshape(len(q),1)
                q_T = q.reshape(1,len(q))
                D = D + (np.dot(p_,p_T) / np.dot(p_T,q_)) - (np.dot(np.dot(np.dot(D,q_),q_T),D)/np.dot(np.dot(q_T,D),q_))
                
                instance[opt].y.append(y.subs(dictx))
                instance[opt].norm.append(norm)
    print("complete")
else:
    print(f"\nThe process took {time() - start}s to complete.\n")


# Tables. If I want the print to be pretty, I only use the pd.DataFrame, without storing it in an object
for n in quests: 
    df = pd.DataFrame(list(zip(instance[n].y,instance[n].norm)),columns=cols)
    print(f"Equation {n}:\n")
    print(df)

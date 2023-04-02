import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


#loi aposteri theta_k sachant theta_k-1
def loi_aposteriori_theta(x,y,theta,theta0,tau0,phi,c,i):
    n=len(y)
    model=LinearRegression()
    model.fit(x,y)
    # calcul de theta
    Xbeta= model.predict(x)
    law = np.exp(-1*(y[i]*np.ones(n-1)-np.array([theta[j] 
                for j in range(len(y)) if j!=i])-np.array([Xbeta[j] 
                for j in range(n) if j!=i]))**2).sum()*(1/len(phi)) + (c/c+len(phi))*np.random.normal(y[i]-theta0,tau0)
    return law


def estim_bayes_theta(x,y,theta0,tau0,phi,c):
    theta=np.zeros(len(y))
    theta_init=theta0*np.ones(len(y))
    for i in range(len(y)):
        theta[i]=loi_aposteriori_theta(x,y,theta_init,theta0,tau0,phi,c,i)
    return theta


# Cette fonction calcule l'estimateur bayésien de beta en se basant sur les loi précédentes
def estim_bayes_beta(y,x,theta0,beta0,tau0,phi,c):
   
    beta_B = np.array([y[i]- estim_bayes_theta(y,x,theta0,tau0,phi,c)[i] 
            for i in range(len(y))]).sum()*np.ones(len(y)) - beta0
                                                                      
    return beta_B


# Cette fonction calcule l'estimateur bayésien de beta en se basant sur les loi précédentes
def estim_bayes_sigma(y,x,theta0,beta0,tau0,phi,c,lamda_0,nu_0):
    n = len(y)
    sigma_B = (np.array([y[i]- estim_bayes_theta(y,x,theta0,tau0,phi,c)[i]- estim_bayes_beta(y,x,theta0,beta0,tau0,phi,c)[i] for i in range(len(y))])).sum()/(nu_0-(n/2)-1)
    return sigma_B
    

def prob_phi(phi,c):
    prob_phi = 1
    for i in range(len(phi)):
        prob_phi = prob_phi*np.math.factorial(len(phi[i])-1)
    return c*prob_phi


def estim_phi_theta(y,x,theta0,beta0,tau0,phi,c):
    theta_phi = np.divide(estim_bayes_theta(x,y,theta0,tau0,phi,c),prob_phi(phi,c)*np.ones(len(y)))
    return theta_phi
    

def estim_phi_beta(y,x,theta0,beta0,tau0,phi,c):
    beta_phi = np.divide(estim_bayes_beta(y,x,theta0,beta0,tau0,phi,c),prob_phi(phi,c)*np.ones(len(y)))
    return beta_phi


def estim_phi_beta(y,x,theta0,beta0,tau0,phi,c):
    beta_phi = np.divide(estim_bayes_beta(y,x,theta0,beta0,tau0,phi,c),prob_phi(phi,c)*np.ones(len(y))) 
    return beta_phi


def estim_phi_sigma(y,x,theta0,beta0,tau0,lamda_0,nu_0,phi,c):
    sigma_phi = estim_bayes_sigma(y,x,theta0,beta0,tau0,lamda_0,nu_0)/prob_phi(phi,c)*np.ones(len(y))
    return sigma_phi


def estim_phi_beta(y,x,theta0,beta0,tau0,phi,c):
    beta_phi = np.divide(estim_bayes_beta(y,x,theta0,beta0,tau0,phi,c),prob_phi(phi,c)*np.ones(len(y))) 
    return beta_phi


def estim_phi_sigma(y,x,theta0,beta0,tau0,lamda_0,nu_0,phi,c):
    sigma_phi = estim_bayes_sigma(y,x,theta0,beta0,tau0,lamda_0,nu_0)/prob_phi(phi,c)*np.ones(len(y))
    return sigma_phi


def  loss_sc(y,x,theta0,beta0,tau0,lamda_0,nu_0,phi,c,k1,k2,k3):
    n=y.shape[0]
    
    SC = (k1/n)*np.linalg.norm(estim_bayes_theta(x,y,theta0,tau0,phi,c)-estim_phi_theta(y,x,theta0,beta0,tau0,phi,c)) 
    + k2*(estim_bayes_sigma(y,x,theta0,beta0,tau0,phi,c,lamda_0,nu_0) - estim_phi_sigma(y,x,theta0,beta0,tau0,lamda_0,nu_0,phi,c))**2 
    + (k3/2)*np.linalg.norm(estim_bayes_beta(y,x,theta0,beta0,tau0,phi,c)-estim_phi_beta(y,x,theta0,beta0,tau0,phi,c))
    
    return SC ,estim_bayes_theta(x,y,theta0,tau0,phi,c)
    

def clustering_2(y,x,theta0,beta0,tau0,lamda_0,nu_0,c,k1,k2,k3,epsilon):
    n = len(y)
    #initialisation 
    phi0=np.array([range(1,n+1)]) # initiation de la partition
    SC0= loss_sc(y,x,theta0,beta0,tau0,lamda_0,nu_0,phi0,c,k1,k2,k3)
    partition_result = phi0  # initialisation de la partition optimale
    SC_result = SC0 # initialisation de la perte optimale
    
    # first step
    
    for i in range(1,n+1):
        phi = np.array([[ x for x in phi0[0] if x!=i],[i]])
        SC = loss_sc(y,x,theta0,beta0,tau0,lamda_0,nu_0,phi,c,k1,k2,k3)
        if SC<= SC_result : 
            partition_result= phi
            SC_result = SC
   
    # second step
    
    if SC_result!=SC0:  # pour éviter le cas où la partition optimale est la partition initiale
        phi1 = partition_result
        SC1 = SC_result
        #j = 0
        for j in range(len(phi1[0])):
        
            phi = np.array([[ x for x in phi1[0] if x!= phi1[0][j]],phi1[1]+[phi1[0][j]]])
            SC = loss_sc(y,x,theta0,beta0,tau0,lamda_0,nu_0,phi,c,k1,k2,k3)
            if abs(SC-SC1)>epsilon:
                if SC<=SC1 :
                    phi1 = phi
                    SC1 = SC
            else: 
                break
        partition_result = phi1 
        SC_result = SC1
        
    return SC_result,partition_result
    
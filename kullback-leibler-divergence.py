#Authors: Agnès Pérez-Millan & Uma Maria Lal-Trehan Estrada
#Date: 07-06-2024
# Cite paper: DOI: https://doi.org/10.21203/rs.3.rs-3982839/v1


import numpy as np
from tqdm import tqdm
import math
from sklearn.neighbors import KernelDensity
import csv
import pandas as pd


## CLASS AND FUNCTIONS TO COMPUTE KL DIVERGENCE AND JS DISTANCE
class HistPdfModel:
    def __init__(self, imgIn, bandwidthIn=2, kernelIn='gaussian'):
        ''''
        Compute PDF of image using Kernel estimator
        '''
        self._kdModel = KernelDensity(bandwidth=bandwidthIn, kernel=kernelIn)
        sample = imgIn.reshape( (len(imgIn.ravel()),1) )
        self._kdModel.fit(sample)

    def getProb(self, domainArr):
        ''''
        Compute probability values given a domain
        return values as array
        '''

        if len(domainArr.shape)==1:
            # convert to column vector to make it compatible with model
            domainArr = domainArr.reshape( (len(domainArr.ravel()),1) )


        probOut = self._kdModel.score_samples(domainArr)
        probOut = np.exp(probOut)
        

        return probOut.ravel()

# calculate the kl divergence
def kl_divergence(p, q):
    s = 0
    for i in range(len(p)):

        if p[i] <= float(1e-10):
            p[i] = float(1e-10)
        if q[i] <= float(1e-10):
            q[i] = float(1e-10)
            
        s = s + p[i] * math.log2(p[i]/q[i])
        
    return s
 
# calculate the js divergence
def js_divergence(p, q):
	m = 0.5 * (p + q)
	return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


## CSV path
#Intorduce the data from FreeSurfer
data_csv = 'dataUnitatALLCS-2023-07-18.csv'
dFr = pd.read_csv(data_csv)


## Check visits
for ii in dFr['ID']:
    if '_02' in ii or '_03' in ii:
        print(ii)


### CORTICAL THICKNESS
## Variables right and Variables left
col_names = dFr.columns
lh_cth = []
rh_cth = []
for col in col_names:
    if 'lh' in col:
        lh_cth.append(col)
    elif 'rh' in col:
        rh_cth.append(col)

# Domain 
domainArr = np.arange( 0, 8 , 0.001)

JSD = []

ind = 0
for subject in tqdm(dFr['ID']):
    
    left_cth = []
    right_cth = []
    for lc in lh_cth:
        left_cth.append((dFr[dFr['ID'] == subject][lc]).values[0])
    for rc in rh_cth:
        right_cth.append((dFr[dFr['ID'] == subject][rc]).values[0])

    rightPdf = HistPdfModel( np.array(left_cth ))
    rightProb = rightPdf.getProb(domainArr)
    leftPdf = HistPdfModel( np.array(right_cth) )
    leftProb = leftPdf.getProb(domainArr)

    js_dist = math.sqrt( js_divergence(rightProb, leftProb) )

    JSD.append(js_dist)

dFr['JSD'] = JSD

#night
import pandas as pd
import numpy as np
import math  
import matplotlib.pyplot as plt
from itertools import groupby
import csv

 
# df = pd.read_csv('MinorP\\MidsemTweets.csv', encoding='utf-8')  # 'utf-8' ensures that emojis are preserved  
 
def findminmax(array):
    print(array.max())
    print(array.min())    
    return array.min(),array.max()
 
def filtertweet(locarray, left_x, right_x, down_y, up_y):
    count = 0
    filtwt = []
    for loc in locarray:
   	 if loc[0]>left_x and loc[0]<right_x and loc[1]>down_y and loc[1]<up_y:
   		 count = count+1
   		 filtwt.append(loc)
#   	  else :  
#   		  unfiltwt.append(loc)
    print(len(filtwt))
    return filtwt
 
#this function find the grid to which the geolocation belong to
def Findcityfromevent(loc,cols, rows):
    x = np.searchsorted(cols,loc[0])
    y = np.searchsorted(rows,loc[1])
    return (y-1)*(len(cols)-1)+(x-1)
 
def ComputeProbabilityDensity(k,X,Y,Epop):
    Zcity= np.zeros([len(Y),len(X)])
    Zpop = np.zeros([len(Y),len(X)])
    llike = 0
    for j in range(len(Y)):
        if j%10==0:
            print(j)
        for i in range(len(X)):
            citynumber = Findcityfromevent((X[i],Y[j]),col_grid_points, row_grid_points)
            Zpop[j,i] = compute_kernel(k,np.array((X[i],Y[j])),Epop)
            Zcity[j,i] = compute_kernel(k,np.array((X[i],Y[j])),train_grids[citynumber])
            llike +=np.log10(balpha[0]*Zcity[j,i]+balpha[1]*Zpop[j,i])
    return Zcity, Zpop, llike/(len(X)*len(Y))

def vectoriseeuclideandistance(e1,E2):
    lon1 = e1[0]
    lat1 = e1[1]
    lon2 = E2[:,0]
    lat2 = E2[:,1]
    radius = 6371 # km
    lon1,lat1,lon2,lat2 = np.radians()
    dlat=np.radians(lat2-lat1)
    dlon=np.radians(lon2-lon1)
    a = np.sqrt(dlat**2+dlon**2)#*PI/180
# 	c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    d = radius * a
    return d #multiplying 1000 makes it in meters ;otherwise it is in kilometers

def vectorisehaversinedistance(e1,E2):
	lon1 = e1[0]
	lat1 = e1[1]
	lon2 = E2[:,0]
	lat2 = E2[:,1]
	radius = 6371 # km
	dlat=np.radians(lat2-lat1)
	dlon=np.radians(lon2-lon1)
	a = (np.sin(dlat/2))**2+ np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*(np.sin(dlon/2))**2
	c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
	d = radius * c
	return d #multiplying 1000 makes it in meters ;otherwise it is in kilometers
 
PI = math.acos(-1)
#CALCULATE DISTANCE IN km
def haversinedistance(lat1,lon1,lat2,lon2):    
    radius = 6371 # km
    dlat=math.radians(lat2-lat1)
    dlon=math.radians(lon2-lon1)
    a = math.sin(dlat/2)*math.sin(dlat/2)+ math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)*math.sin(dlon/2)
    c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d = radius * c
    return d
 
#distance measure between two geo-locations
def distance(e1, e2):
    lon1 = e1[0]
    lat1 = e1[1]
    lon2 = e2[0]
    lat2 = e2[1]
    return 1000*haversinedistance(lat1,lon1,lat2,lon2) #multiplying 1000 makes it in meters ;otherwise it is in kilometers
 
#brute force implementation of k-NN search
def find_kth_nearest_neighbour(k, e, E):
    dist = []
    # for i in range(len(E)):
    #    	 dist.append(distance(e, E[i]))
#         dist = vectorisehaversinedistance(e,E)
#         dist.sort()
    return dist[k - 1]
 
def vector_find_kth_nearest_neighbour(k, dist):
#     	dist = []
# 	# for i in range(len(E)):
# 	#    	 dist.append(distance(e, E[i]))
#     	dist = vectorisehaversinedistance(e,E)
    	dist.sort()
	#     print(dist)
    	return dist[k - 1]
 
 
def compute_kernel(k, e, E):

    prob_den = 1e-12
    if k<=len(E):
        vectordistance = vectorisehaversinedistance (e,E) #used to find prob_den
        vectordistance2 = vectordistance #used to sort
        # h = spread ie nearest neighbour distance
        h = (vector_find_kth_nearest_neighbour (k, vectordistance2))
        if h!=0:
        #     print(h)
            n = len(E)
            prob_den = 0
            PI = math.acos(-1)
            prob_den = np.sum(np.exp(-((vectordistance) ** 2) / (2.0 * (h))))
            ans = prob_den/(2 * PI * (h) * n)
            return ans
    return prob_den
 
#val = validation set events, tra = training set events,  
def compute_f1_f2(k):
    f2 = []
    f1 = []
    for i in range(len(val)):
        if i%1000==0:
            print(i)
        f2.append(compute_kernel(k, val[i], training))
        f1.append(compute_kernel(k, val[i], train_grids[val_grid[i]]))
    return f1, f2
 
#as there are only 2 parameter, we are finding the best parameter through grid search
def best_parameters(f1, f2):
        best_alpha = (1.000, 0.000)
        best_value = 0
        loglikelihood_list = []
        n = len(f1)
        best_value = np.sum(np.log10(f1))
        best_value /= n
        loglikelihood_list.append((1.000, best_value))
        for i in range(1, 1001):  
            alpha = ((1000 - i) / 1000.0, i / 1000.0)
            value = 0
            value = np.sum(np.log10( alpha[0]*f1+alpha[1]*f2))
            value /= n
    #     	for j in range(n):
    #         	value += math.log(alpha[0] * f1[j] + alpha[1] * f2[j])
            loglikelihood_list.append((alpha[0], value))
            if value > best_value:
                best_value = value
                best_alpha = alpha
        return best_alpha, best_value, loglikelihood_list


 
# ('NV', 'Nevada', '35.003', '42.0003', '-120.0037', '-114.0436')
 
left_x , right_x = -117, -114.036 #longitude #depend on us
down_y , up_y = 35.003, 36.5 #latitude #depend on us
 
bottomLeft = (left_x, down_y)  
bottomRight = (right_x, down_y)
topLeft = (left_x, up_y)  
topRight = (right_x, up_y)  
 
sqrtcity=5 #parameter
grids = sqrtcity**2  
col_grid_points = np.linspace(bottomLeft[0], bottomRight[0], num=sqrtcity+1) #sqrtcity number of grids
row_grid_points = np.linspace(bottomLeft[1], topLeft[1], num=sqrtcity+1) #sqrtcity number of grids



df = pd.read_csv('Nevada_Lat_long_globalpoints.csv', encoding='utf-8')  # 'utf-8' ensures that emojis are preserved  
df = df.values#making it as a array
df = df[:,1:]# removing the first column as it contains numbers
 
filtwts = filtertweet(df, left_x, right_x, down_y, up_y)
filtwts = np.unique(filtwts,axis=0)  
print("np.unique",len(filtwts))

allresult = []
for k in range(3,9,2):
    print("k=",k)
    np.random.shuffle(filtwts)#randomised filtered tweets

    #converted to array
    filtwts = np.array(filtwts)

    #few starting tweets into train set and next few tweets into validation set
    splitid = 10000
    training, val, test = filtwts[:splitid,:], filtwts[splitid:2*splitid,:], filtwts[2*splitid:3*splitid,:]

    #training set assigned a grid number
    box = []
    for loc in training:
        box.append(Findcityfromevent(loc,col_grid_points, row_grid_points))

    #classifying location to each grid
    train_grids = [[] for i in range(grids)]
    for loc in training:
        train_grids[Findcityfromevent(loc,col_grid_points, row_grid_points)].append(loc)


    train_grids = [np.array(i) for i in train_grids]

    # print()
    print(len(training))

    val_grid = []
    for loc in val:
        val_grid.append(Findcityfromevent(loc,col_grid_points, row_grid_points))

    print("finding probability distribution grid wise")
    # what should be the k value, and how should we find the log of 0
    ff1, ff2 = compute_f1_f2(k)
    ff1 = np.array(ff1)
    ff2 = np.array(ff2)
    #print("f1=",ff1,"\n","f2=",ff2)
    balpha, bvalue, llist = best_parameters(ff1, ff2)
    result = []
    result.append((k,ff1,ff2,balpha,bvalue,llist))
    allresult.append(result)
    print(balpha, bvalue)
    llist = np.array(llist)
    plt.plot(llist[:,0], llist[:,1])
#     plt.title('')
    plt.xlabel('Weights for city')
    plt.ylabel('Log-Likelihood')
    plt.show()
    epsilon = 0.0000001
    x = np.linspace(left_x+epsilon, right_x-epsilon, 50) #lon
    y = np.linspace(down_y+epsilon,up_y-epsilon, 50)   #lat
    numberofcity = sqrtcity**2
    X, Y = np.meshgrid(x, y)
    Zcity, Zpop, llikelihood = ComputeProbabilityDensity(5,x,y,training)
#     print("testllike", llikelihood)
    Z = balpha[0]*Zcity + balpha[1]*Zpop
    # plt.scatter(training[:,0],training[:,1] , c='k', alpha=0.9, s=3)
    plt.contour(x,y,Z,100,color='black')
    plt.colorbar()
    plt.show()
    
    test_grid = []
    for loc in test:
        test_grid.append(Findcityfromevent(loc,col_grid_points, row_grid_points))
    
    ff1, ff2 = compute_f1_f2(k)
    ff1 = np.array(ff1)
    ff2 = np.array(ff2)
    testlikelihood = np.mean(np.log10( balpha[0]*ff1+balpha[1]*ff2))
    print("testlikelihood", testlikelihood)

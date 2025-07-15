import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import matplotlib.patches as patches
import math
import scipy.sparse as sps
from scipy.ndimage import gaussian_filter
from sklearn.feature_selection import mutual_info_regression as mi_skl
import seaborn as sns
from scipy import stats
import time
from matplotlib import gridspec

def sandwell_loc(video, fta = 400):
    print(video)
    cap = cv2.VideoCapture(video)
    sandwell_status = 'n'
    fta = fta
    while sandwell_status != 'y':
        for i in list(np.arange(1,3000,fta)):
            cap.set(1, i)
            ret, img = cap.read()
            img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            img = cv2.addWeighted(img1,(i/(i+1)),img1,(1/(i+1)),0)

        img = cv2.addWeighted(img,.8,img,.8,0)
        img[:,:270]=cv2.addWeighted(img[:,:270],0,img[:,:270],0,0)
        img[:,1150:]=cv2.addWeighted(img[:,1150:],0,img[:,1150:],0,0)
        gray_blurred = cv2.blur(img, (3, 3)) 

        sandwells = cv2.HoughCircles(gray_blurred,  
                                   cv2.HOUGH_GRADIENT, 1, 200, param1 = 250, 
                                   param2 = 25, minRadius = 14, maxRadius = 20)
        # Convert the circle parameters a, b and r to integers. 
        sandwells = np.uint16(np.around(sandwells))  
  
        if len(sandwells[0])==6:
            for pt in sandwells[0, :]: 
                    x, y, r = pt[0], pt[1], pt[2]
                    cv2.circle(img1, (x, y), r+10, (255, 0,0), 2)
            
            plt.figure(figsize=(4,4))
            plt.imshow(img1)
            plt.show()
            sandwell_status = input("Are all sandwells correct - y/n?")
            fta += 20
            
        else:
            fta+=20
    return sandwells, img



def get_corners(video):
    corners = [0]
    fta = 400
    while len(corners) < 8:
        sandwells, img = sandwell_loc(video, fta)
        step =155
        y_min= min(sandwells[0,:][:,1])-step
        y_max= max(sandwells[0,:][:,1])+step
        x_min= min(sandwells[0,:][:,0])-step
        x_max= max(sandwells[0,:][:,0])+step

        img_cropped=img[y_min:y_max, x_min:x_max]
        img_gray = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)    
        img_cropped = cv2.blur(img_cropped, (8, 8)) 

        mask = np.ones(img_cropped.shape)
        mask[0:,100:650] = 0; mask[100:650:,0:] = 0
        mask = cv2.cvtColor(np.uint8(mask), cv2.COLOR_GRAY2BGR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(img_cropped, 8, 0.1, 40, mask = mask, blockSize = 8)   
        corners = np.int0(corners)
        fta+=20
        
    for i in corners:
        x,y = i.ravel()
        img_gray=cv2.circle(img_gray,(x+x_min,y+y_min),5,(0,255,0),-1)

    for pt in sandwells[0, :]: 
            x, y, r = pt[0], pt[1], pt[2]
            cv2.circle(img_gray, (x, y), r+10, (255, 0,0), 2)
    
    Nx = [600,800]; Ny = [0,200]; Nwx = Nx[1]-Nx[0]; Nwy = Ny[1]-Ny[0]
    Ex = [200,270]; Ey = [590,750]; Ewx = Ex[1]-Ex[0]; Ewy = Ey[1]-Ey[0]
    Sx = [600,800]; Sy = [850,1100]; Swx = Sx[1]-Sx[0]; Swy = Sy[1]-Sy[0]
    Wx = [1100,1275]; Wy = [300,450]; Wwx = Wx[1]-Wx[0]; Wwy = Wy[1]-Wy[0]
    
    plt.figure(figsize=(2,2))
    
    plt.imshow(img_gray, cmap='gray')
    rectN = patches.Rectangle((Nx[0], Ny[0]), Nwx, Nwy, linewidth=0.2, edgecolor='r', fc='none')
    rectE = patches.Rectangle((Ex[0], Ey[0]), Ewx, Ewy, linewidth=0.2, edgecolor='r', fc='none')
    rectS = patches.Rectangle((Sx[0], Sy[0]), Swx, Swy, linewidth=0.2, edgecolor='r', fc='none')
    rectW = patches.Rectangle((Wx[0], Wy[0]), Wwx, Wwy, linewidth=0.2, edgecolor='r', fc='none')
    plt.show()

    new_corners=np.array([corners.item(i) for i in range(len(corners)*2)]).reshape(8,2)
    new_corners[:,0]=new_corners[:,0]+x_min
    new_corners[:,1]=new_corners[:,1]+y_min
            
    return sandwells[0][:,:2], new_corners

def arena_coodinates(BEHvid):
    #Extract sandwells & corners coordinates
    sandwells, corners = get_corners(BEHvid)
    cx = corners[:,0]
    cy = corners[:,1]
    
    
    bl_x = min(cx); bl_y = min(cy) #bl = bottom left corner
    tr_x = max(cx); tr_y = max(cy) #tr = top right corner
    w_x = tr_x - bl_x;  w_y = tr_y - bl_y #w = width
    
    cxn_x = bl_x #correction factor x
    cxn_y = bl_y #correction factor y
    
    #correct corners & sws
    bl_x -= cxn_x; bl_y -= cxn_y
    tr_x -= cxn_x; tr_y -= cxn_y
    w_x = tr_x - bl_x; w_y = tr_y - bl_y
    sw_x = sandwells[:,0] - cxn_x; sw_y = sandwells[:,1] - cxn_y

    return bl_x, bl_y, tr_x, tr_y, w_x, w_y, sw_x, sw_y, cxn_x, cxn_y


def direction_angle(row, df):
    dx = df.loc[row,'leftear_x'] - df.loc[row,'rightear_x']
    dy = df.loc[row,'rightear_y'] - df.loc[row,'leftear_y']
    angle = math.atan2(dy,dx)
    #angle=math.degrees(angle)
    angle=math.pi-angle
    return (math.pi*2 + angle) if angle < 0 else angle, math.degrees((math.pi*2 + angle)) if angle < 0 else math.degrees(angle)

def arena_binned(x, y, xdim, ydim, pix_cm, pf_cm):
    """Creates occupancy maps by bibning arena counting bin occupancy
    INPUTS
    ------
    x = x pixel coordinates, 
    y = y pixel coordinates
    xdim = pixel coordinates for the arena sides e.g. [0, 500]
    ydim = pixel coordinates for the arena top and bottom e.g. [0, 500],
    pix_cm = conversion of number of pixels to cm
    pf_cm = size of the bins to use (cm)
    
    RETURNS:
    ------
    linS = Linearised bin occupancy
    occMap = 2D matrix of occupancy values
    nBnx = number of bins in x
    nBny = number of bins in y 
    bx = bin edges x
    by = bin edges y
    """
    
    d_x = xdim[0] - xdim[1] # xdim[0] = max, xdim[1] = min
    d_y = ydim[0] - ydim[1] # ydim[0] = max, ydim[1] = min
    nBnx = int((d_x/(pix_cm*pf_cm)))
    nBny = int((d_y/(pix_cm*pf_cm)))
    bn_x = [int(i) for i in np.linspace(xdim[0],xdim[1],nBnx)]
    bn_y = [int(i) for i in np.linspace(ydim[0], ydim[1],nBny)]

    bx = ((x-bn_x[0])/(bn_x[1]-bn_x[0])).astype(int)
    by = ((y-bn_y[0])/(bn_y[1]-bn_y[0])).astype(int)

    S = np.vstack((bx,by))
    linS = np.ravel_multi_index(S,(nBnx,nBny))
    occMap = sps.csr_matrix((np.ones(len(bx)),(bx,by)),shape=(nBnx,nBny),dtype=float).todense()
    return linS.astype(float), occMap, nBnx, nBny, bx, by


def bursting_check(linSpf, linS, e_trace, tr_delay):
    """Checks if the cell randomly bursts in a bin (which may equate to high mutual information)
    or if it is firing consistently across bin visits
    
    INPUTS
    ------
    linSpf = linearised firing in each bin
    linS = linearised occupancy in each bin
    e_trace = event traces
    tr_delay = the delay between bin entries to consider (e.g if the cell is bursting 
    and the animals leaves a bin for 1 second, then returns it may be mistinterpreted as a place cell  )
    
    RETURNS:
    tr = the number of times the animal traverses that bin
    rdm_burst = the number of traversals that the cell fires
    """
    #Place field timestamps
    pft_idxs = [i for i in range(len(linS)) if linS[i] in linSpf]
  
    #Seperate timestamps into traverals
    tr_idxs_all = []; tr_idxs = []
    i = 0
    while i < len(pft_idxs)-1:
        tr = pft_idxs[i+1]-pft_idxs[i]
        tr_idxs.append(pft_idxs[i])

        if tr > tr_delay: #acceptable delay between traversals
            tr_idxs_all.append(tr_idxs)
            tr_idxs = []
        i += 1
    
    if len(tr_idxs_all) > 0: #If 1 single traversal
        
        #Calculate the number of traversals with events
        tr = 0
        for i in tr_idxs_all:
            n_events = sum(e_trace[i])
            if n_events > 0:
                tr += 1
    else:
        tr = 1; tr_idxs_all = [[0]]
    
    return tr,(tr/len(tr_idxs_all))*100

def arena_binned(x, y, xdim, ydim, pix_cm, pf_cm):
    """Creates occupancy maps by bibning arena counting bin occupancy
    INPUTS
    ------
    x = x pixel coordinates, 
    y = y pixel coordinates
    xdim = pixel coordinates for the arena sides e.g. [0, 500]
    ydim = pixel coordinates for the arena top and bottom e.g. [0, 500],
    pix_cm = conversion of number of pixels to cm
    pf_cm = size of the bins to use (cm)
    
    RETURNS:
    ------
    linS = Linearised bin occupancy
    occMap = 2D matrix of occupancy values
    nBnx = number of bins in x
    nBny = number of bins in y 
    bx = bin edges x
    by = bin edges y
    """
    
    d_x = xdim[0] - xdim[1] # xdim[0] = max, xdim[1] = min
    d_y = ydim[0] - ydim[1] # ydim[0] = max, ydim[1] = min
    nBnx = int((d_x/(pix_cm*pf_cm)))
    nBny = int((d_y/(pix_cm*pf_cm)))
    bn_x = [int(i) for i in np.linspace(xdim[0],xdim[1],nBnx)]
    bn_y = [int(i) for i in np.linspace(ydim[0], ydim[1],nBny)]

    bx = ((x-bn_x[0])/(bn_x[1]-bn_x[0])).astype(int)
    by = ((y-bn_y[0])/(bn_y[1]-bn_y[0])).astype(int)

    S = np.vstack((bx,by))

    linS = np.ravel_multi_index(S,(nBnx,nBny))
    occMap = sps.csr_matrix((np.ones(len(bx)),(bx,by)),shape=(nBnx,nBny),dtype=float).todense()
    return linS.astype(float), occMap, nBnx, nBny, bx, by

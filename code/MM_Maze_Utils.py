import os
import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from copy import deepcopy
from dataclasses import make_dataclass
from MM_Plot_Utils import plot

# Maze-related routines

Maze = make_dataclass('Maze', ['le','ru','pa','ch','xc','yc','ce','rc','di','cl','wa','st'])
    # A structure to hold the definition of a maze. 
    # See help(NewMaze) for docstring.

def NewMaze(n=6): # n must be even for square maze
    '''
    This constructs a binary maze with n branches according to our standard design.
    n must be even to get a square maze.
    The maze consists of square cells the width of a corridor.
    Each cell in the maze has an (x,y) location, with x,y = 0,...,2**(n/2+1)-2.
    y runs top to bottom.
    Each cell is also given a unique integer cell number.
    The present maze consists of straight runs, each run terminates in a branch point or 
    an end point. 
    Each run has a parent run from which it branched off (except the entry run).
    Each run has two children that branch from it (except the end runs)
    This routine constructs the list of cells in each run and a lot of other useful
    data to operate on the maze.
    Data fields:
    le = Number of levels; int.
    ru = Runs. List of lists of cell numbers.
    pa = Parent runs. Index of the parent of each run; (nr) array.
    ch = Child runs. Indexes of the two children of each run; (nr,2) array.
    rc = Run containing a given cell; (nc) array.
    xc = x-coordinate for a given cell. (nc) array.
    yc = y-coordinate for a given cell. (nc) array.
    di = Distance for a given pair of cells. (nc,nc) array.
    ce = cell number for a given (x,y) tuple; dict.
    cl = centerline (x,y) positions for drawing the maze; (n,2) array.
    wa = wall (x,y) positions for drawing the maze; (n,2) array.
    st = array identifying types of steps between nodes; (nr.nr) array.
    '''
    ru = [] # runs, each is a list of cells that ends in a branch point or leaf
    pa = [] # parent run index
    for i in range(n+1): # i is the level of the binary tree, with the root branch point at i=0
        xd = (i+1)%2 # whether to step in the x direction from the parent run endpoint; this depends only on level
        yd = i%2 # whether to step in the y direction from the parent run endpoint
        di = 2**((n-i)//2) # length of the run
        for j in range(2**i): # j is the index of the branch points within a level
            if i==0: # this is the entry run that terminates at the first branch point
                k = -1 # entry run has no parent
                pa.append(k)
                (x,y) = (int(2**(n/2)-1),int(2**(n/2)-1)) # end point at the center of the maze
                ru.append([(x1,y) for x1 in range(0,x+1)]) # straight run from x=0 to the center
            else:
                k = 2**(i-1)-1+j//2 # parent run is one level up in the tree
                pa.append(k)
                (x0,y0) = ru[k][-1] # end point of parent run
                (xs,ys) = (xd*(2*(j%2)-1),yd*(2*(j%2)-1)) # single step size along x and y; can be +1, -1, or 0
                (x,y) = (x0+xs*di,y0+ys*di) # endpoint of current run
                if xs==0: # if run goes in y direction
                    ru.append([(x,y1) for y1 in range(y0+ys,y+ys,ys)])
                else: # if run goes in x direction
                    ru.append([(x1,y) for x1 in range(x0+xs,x+xs,xs)])                    
    ce = {} # dictionary of cell number for a given (x,y) location
    lo = {} # dictionary of (x,y) location for a given cell number 
    c = 0
    for r in ru: # assign the cell numbers to locations along the runs
        for p in r:
            ce[p] = c
            lo[c] = p
            c += 1
    nc = c # number of cells
    ru = [[ce[p] for p in r] for r in ru] # convert the runs from (x,y) locations to cell numbers
    pa = np.array(pa)
    ch = np.full((len(ru),2),-1,dtype=int)
    for i,p in enumerate(pa):
        if p>=0: # -1 indicates a run with no parent
            if ch[p,0]==-1: # if the first child hasn't been found
                ch[p,0]=i
            else:
                ch[p,1]=i
    xc = np.array([lo[c][0] for c in range(nc)])
    yc = np.array([lo[c][1] for c in range(nc)])
    ma = Maze(le=n,ru=ru,pa=pa,ch=ch,xc=xc,yc=yc,ce=ce,rc=None,di=None,cl=None,wa=None,st=None)
    ma.rc = np.array([RunIndex(c,ma) for c in range(nc)])
    ma.di = ConnectDistance(ma)
    ma.cl = MazeCenter(ma)
    ma.wa = MazeWall(ma)
    ma.st = MakeStepType(ma)
    return ma

def RunIndex(c,m):
    '''
    Returns the index of the run that contains the cell c in maze m
    '''
    for i,r in enumerate(m.ru):
        if c in r:
            return i

def HomePath(c,m):
    '''
    Returns a path that leads from cell c to the start of the maze m
    Includes both start and end cells
    '''
    ret = []
    i = m.rc[c] # index of run containing c
    ret+=m.ru[i][m.ru[i].index(c)::-1] # reverse that run starting at c
    i = m.pa[i] # find the parent run
    while i>=0:
        ret+=m.ru[i][::-1] # add the reversed parent run
        i = m.pa[i] # look for its parent
    return ret
    
def HomeDistance(m):
    '''
    Returns an array that gives for every cell c the distance from the starting of the maze m
    '''
    di = np.zeros(len(m.xc)) # number of cells in the maze
    for r in m.ru:
        for c in r:
            di[c]=len(HomePath(c,m))-1
    return di
    
def ConnectPath(c1,c2,m):
    '''
    Returns the shortest path that connects cells c1 and c2 in maze m
    Includes both start and end cells
    '''
    r1 = HomePath(c1,m)
    r2 = HomePath(c2,m)[::-1] # reversed
    for i in r1:
        if i in r2:
            return (r1[:r1.index(i)]+r2[r2.index(i):])

def ConnectDistance(m):
    '''
    Returns a 2D array that gives the distance for every pair of cells in maze m.
    This is the smallest number of steps to reach one cell from the other
    '''
    nc = len(m.xc) # number of cells in the maze
    di = np.array([[len(ConnectPath(c1,c2,m))-1 for c2 in range(nc)] for c1 in range(nc)])
    return di

def MakeStepType(m):
    '''
    Makes an accessory array that tells for a pair of successive nodes 
    whether the step was in left (0), in right (1), out left (2), or out right (3).
    "in" means into the maze, taking the left or right branch of the T junction.
    "out left" means out of the maze along the "left" branch as seen from the 
    parent T junction.
    '''
    exitstate=len(m.ru) # 1 + highest node number
    st = np.full((len(m.ru)+1,len(m.ru)+1),-1,dtype=int) # accessory array
    for i in range (m.le+1): # level 0 is the first branch point, node 0
        for j in range (2**i-1,2**(i+1)-1):
            if j>0: # first node has no 'out' steps
                if (i+j+m.pa[j])%2 == 0:
                    st[j,m.pa[j]]=2 # 'out left'=2
                else:
                    st[j,m.pa[j]]=3 # 'out right'=3
            if i<m.le: # last level has no 'in' steps
                for c in m.ch[j]:
                    if (i+j+c)%2 == 0:
                        st[j,c]=1 # 'in right'=1
                    else:
                        st[j,c]=0 # 'in left'=0
    st[0,exitstate]=3 # special case of exit from node 0
    return st 

def StepType(i,j,m):
    '''
    Returns the type of step from node i to j in maze m.
    in left = 0; in right = 1; out left = 2; out right = 3; illegal = -1
    "in" means into the maze, taking the left or right branch of the T junction.
    "out left" means out of the maze along the "left" branch as seen from the 
    parent T junction.
    '''
    return m.st[int(i),int(j)]

def StepType2(i,j,m):
    '''
    A version of StepType() that considers both 'out left and 'out right' steps the same.
    Returns the type of step from node i to j in maze m.
    in left = 0; in right = 1; out = 2; illegal = -1
    "in" means into the maze, taking the left or right branch of the T junction.
    "out" means out of the maze along the stem of the T junction.
    '''
    st2 = m.st[int(i),int(j)]
    if st2==3: # make both 'out left' and 'out right' = 2
        st2=2
    return st2

def StepType3(i,j,m):
    '''
    A version of StepType() that considers both 'in left and 'in right' steps the same.
    Returns the type of step from node i to j in maze m.
    in = 0; out left = 1; out right = 2; illegal = -1
    "in" means into the maze, taking the left or right branch of the T junction.
    "out" means out of the maze along the stem of the T junction.
    '''
    st3 = m.st[int(i),int(j)]
    if st3==0 or st3==1: # make both 'in left' and 'in right' = 0
        st3=0
    elif st3==2 or st3==3: # make 'out left' = 1 and 'out right' = 2
        st3-=1
    return st3

def MazeCenter(m):
    '''
    Returns an nx2 array of (x,y) values that represents the centerline of the maze
    '''
    def acc(i): # accumulates a path through the cells of run i and all its children and back
        r = m.ru[i][:]
        if m.ch[i,0]!=-1:
            r += acc(m.ch[i,0])
            r += [m.ru[i][-1]]
            r += acc(m.ch[i,1])
            r += m.ru[i][-1::-1]
        return r
    c = acc(0)
    return np.array([m.xc[c],m.yc[c]]).T

def PlotMazeCenter(m,axes=None,numbers=False):
    '''
    Plot the maze defined in m, draws a stick figure of the centerline of each corridor
    axes: provide this to add to an existing plot
    numbers: sets whether the cells are numbered
    '''
    w = MazeCenter(m)    
    if axes:
        plot(w[:,0],w[:,1],fmts=['r-'],equal=True,linewidth=1,figsize=(6,6),yflip=True,axes=axes) # this way we can add to an existing graph
    else:
        ax = plot(w[:,0],w[:,1],fmts=['r-'],equal=True,linewidth=1,figsize=(6,6),yflip=True)
    if numbers:
        for c in range(len(m.xc)):
            plt.text(m.xc[c],m.yc[c],'{:d}'.format(c)) 
            
def MazeWall(m):
    '''
    Returns an nx2 array of (x,y) values that represents the walls of the maze
    '''
    (xc,yc,ru,pa) = (m.xc,m.yc,m.ru,m.pa)
    ch = [np.where(np.array(pa)==i)[0].astype(int) for i in range(len(ru))] # array of all children of runs
    
    def acw(i): # recursive function that returns a path for the wall starting with run i
        r = ru[i] 
        c0 = np.array([xc[r[0]],yc[r[0]]]) # first cell in this run
        c1 = np.array([xc[r[-1]],yc[r[-1]]]) # last cell in this run
        if i==0:
            d = np.array([1,0]) # direction of the entry run
        else:
            p1 = np.array([xc[ru[pa[i]][-1]],yc[ru[pa[i]][-1]]]) # last cell of parent run
            d = c0-p1 # direction of this run
        sw = 0.5*np.array([-d[0]-d[1],d[0]-d[1]]) # diagonal displacements
        se = 0.5*np.array([-d[0]+d[1],-d[0]-d[1]]) # compass directions for a run pointing (0,1)               
        nw = 0.5*np.array([d[0]-d[1],d[0]+d[1]])              
        ne = 0.5*np.array([d[0]+d[1],-d[0]+d[1]])
        if i==0:
            p = [c0+sw] # start point
        else:
            p = []
        p += [c1+sw] # to end of this run on left side
        if len(ch[i]):
            e = np.array([xc[ru[ch[i][0]][0]],yc[ru[ch[i][0]][0]]])-c1 # direction of the first child path
            if np.array_equal(e,np.array([-d[1],d[0]])): # w direction, i.e. left
                il = ch[i][0]; ir = ch[i][1] # determine left and right child paths
            else:
                il = ch[i][1]; ir = ch[i][0]
            p += acw(il) # accumulate the left path
            p += [c1+ne] # short connector on far side
            p += acw(ir) # accumulate right path
            p += [c0+se]  # finish the reverse path  
        else: # an end point
            p += [c1+nw, c1+ne, c1+se] # go around the endpoint
        return p
    return np.array(acw(0))

def PlotMazeWall(m,axes=None,figsize=4):
    '''
    Plots the walls of the maze defined in m.
    axes: provide this to add to an existing plot
    figsize: in inches (only if axes=None)
    '''
    if axes:
        plot(m.wa[:,0],m.wa[:,1],fmts=['k-'],equal=True,linewidth=2,
             xhide=True,yhide=True,axes=axes) # this way we can add to an existing graph
    else:
        axes = plot(m.wa[:,0],m.wa[:,1],fmts=['k-'],equal=True,linewidth=2,yflip=True,
                  figsize=(figsize,figsize),xhide=True,yhide=True)
    return axes

def PlotMazeNums(m,axes,mode='cells',numcol='blue'):
    '''
    adds numbering to an existing maze plot given by axes
    m: maze
    mode: 'cells','runs','nodes': depending on what gets numbered
    numcol: color of the numbers
    '''
    if mode=='nodes':
        for j,r in enumerate(m.ru):
            x = m.xc[r[-1]]; y=m.yc[r[-1]]
            plt.text(x-.35,y+.15,'{:d}'.format(j),color=numcol) # number the ends of a run    
    if mode=='cells':
        for j in range(len(m.xc)):
            x = m.xc[j]; y=m.yc[j]
            plt.text(x-.35,y+.15,'{:d}'.format(j),color=numcol) # number the cells
    if mode=='runs':
        for j,r in enumerate(m.ru):
            xlo = min(m.xc[r]); xhi = max(m.xc[r]); ylo = min(m.yc[r]); yhi = max(m.yc[r])
            ax.add_patch(patches.Rectangle((xlo-0.5,ylo-0.5),xhi-xlo+1,yhi-ylo+1,lw=1,fill=False)) 
            x = 0.5*(m.xc[r[0]]+m.xc[r[-1]])-0.35; y = 0.5*(m.yc[r[0]]+m.yc[r[-1]])+0.15
            plt.text(x,y,'{:d}'.format(j),color=numcol)             

def PlotMazeFunction(f,m,mode='cells',numcol='cyan',figsize=4,col=None,axes=None):
    '''
    Plot the maze defined in m with a function f overlaid in color
    f[]: array of something as a function of place in the maze, e.g. cell occupancy
        If f is None then the shading is omitted
    m: maze structure
    mode: 'cells','runs','nodes': depending on whether f[] is associated with either of these
    numcol: color for the numbers. If numcol is None the numbers are omitted
    figsize: in inches
    col: a color scale to map f[] into colors. nx4 ndarray. 
        col[j,0]=value of f associated with this color
        col[j,1:4]=rgb values of this color, each in the range [0,1]
        if col==None then the color scale is from 0=white to 1=black
    Returns: the axes of the plot.
    '''		
    if col is None:
        col=np.array([[0,1,1,1],[1,0,0,0]]) # default 0=white to 1=black 

    def Color(x): 
        return [np.interp(x,col[:,0],col[:,j]) for j in [1,2,3]]

    if axes:
        ax=axes
        PlotMazeWall(m,axes=ax,figsize=figsize)
    else:
        ax=PlotMazeWall(m,axes=None,figsize=figsize)

    if mode=='nodes':
        for j,r in enumerate(m.ru):
            x = m.xc[r[-1]]; y=m.yc[r[-1]]
            if not(f is None):
                ax.add_patch(patches.Rectangle((x-0.5,y-0.5),1,1,lw=0,
                                           color=Color(f[j]))) # draw with color f[]
            if numcol:
                plt.text(x-.35,y+.15,'{:d}'.format(j),color=numcol) # number the ends of a run    
    if mode=='cells':
        for j in range(len(m.xc)):
            x = m.xc[j]; y=m.yc[j]
            if not(f is None):
                ax.add_patch(patches.Rectangle((x-0.5,y-0.5),1,1,lw=0,
                                           color=Color(f[j]))) # draw with color f[]
            if numcol:
                plt.text(x-.35,y+.15,'{:d}'.format(j),color=numcol) # number the cells
    if mode=='runs':
        for j,r in enumerate(m.ru):
            xlo = min(m.xc[r]); xhi = max(m.xc[r]); ylo = min(m.yc[r]); yhi = max(m.yc[r])
            if not(f is None):
                ax.add_patch(patches.Rectangle((xlo-0.5,ylo-0.5),xhi-xlo+1,yhi-ylo+1,lw=0,
                                          color=Color(f[j])))  # draw with color f[]
            else:
                ax.add_patch(patches.Rectangle((xlo-0.5,ylo-0.5),xhi-xlo+1,yhi-ylo+1,lw=1,
                                          color='black',fill=False))  # draw outline
            if numcol:
                x = 0.5*(m.xc[r[0]]+m.xc[r[-1]])-0.35; y = 0.5*(m.yc[r[0]]+m.yc[r[-1]])+0.15 
                plt.text(x,y,'{:d}'.format(j),color=numcol) # number the middle of a run             
    return ax

def PlotMazeCells(m,numcol='blue',figsize=6):
    '''
    Plots the maze wall and numbers the cells
    numcol: color for the numbers. If numcol is None the numbers are omitted
    figsize: in inches
    '''
    PlotMazeFunction(None,m,mode='cells',numcol=numcol,figsize=figsize,col=None)
    
def PlotMazeRuns(m,numcol='blue',figsize=6):
    '''
    Plots the maze wall and numbers the runs
    numcol: color for the numbers. If numcol is None the numbers are omitted
    figsize: in inches
    '''
    PlotMazeFunction(None,m,mode='runs',numcol=numcol,figsize=figsize,col=None)
    
def PlotMazeNodes(m,numcol='blue',figsize=6):
    '''
    Plots the maze wall and numbers the nodes
    numcol: color for the numbers. If numcol is None the numbers are omitted
    figsize: in inches
    '''
    PlotMazeFunction(None,m,mode='nodes',numcol=numcol,figsize=figsize,col=None)
    
def NodeLevel(n):
    '''
    Returns the level of node n
    '''
    return int(np.floor(np.log(n+1)/np.log(2)))
    

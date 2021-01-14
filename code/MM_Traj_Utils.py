import pickle
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from copy import deepcopy
from dataclasses import make_dataclass
from MM_Plot_Utils import plot
from MM_Maze_Utils import *

# Trajectory routines 
Traj = make_dataclass('Traj', ['fr','ce','ke','no','re']) # a simple structure for trajectory data 
Traj.__module__ = __name__ # for some reason this is needed for pickling the structure

def NewTraj(fr=None,ce=None,ke=None,no=None,re=None):
    '''
    Returns a Traj structure with the desired components
    fr: start and end frame for each bout; (n,2) ndarray 
    ce: cell number in each frame; list of ndarrays, one for each bout
    ke: nose keypoint (x,y) in each frame; list of (n,2) ndarrays, one for each bout
    no: node number and start frame within the bout; list of (n,2) ndarrays, one for each bout
    re: start frame and end frame within the bout for each reward; list of (n,2) ndarrays, one for each bout
    '''
    return Traj(fr,ce,ke,no,re)

def TestTrajModule():
	return 'OK'

def LoadTraj(filename):
    '''
    Loads a trajectory from a pickle file.
    Puts the results in a Traj structure.
    '''
    with open('outdata/'+filename, 'rb') as f:
        return pickle.load(f)

def SaveTraj(filename,tr):
    '''
    Saves a trajectory to a pickle file.
    Puts the results in a Traj structure.
    '''
    with open('outdata/'+filename, 'wb') as f:
        pickle.dump(tr, f)

def PlotTraj(bo,tr,ma):
    '''
    Plot bout bo of trajectory tr in maze ma
    Both continuous and discrete positions
    Use an approximate scaling that seems to align them
    '''
    ce = tr.ce[bo]
    ce1 = ce[ce>=0] # eliminate the bad frames where ce=-1
    if tr.ke: # check if continuous keypoint data are included; plot these first
        ax = plot(-0.5+15*tr.ke[bo][:,0],-0.5+15*tr.ke[bo][:,1],
                  fmts=['r-'],linewidth=1,figsize=(4,4),equal=True,yflip=True,
                  xlim=[-1,15],ylim=[-1,15]);
        plot(ma.xc[ce1],ma.yc[ce1],equal=True,linewidth=1,axes=ax);
    else:
        plot(ma.xc[ce1],ma.yc[ce1],equal=True,linewidth=1,figsize=(4,4),yflip=True,
            xlim=[-1,15],ylim=[-1,15]);
        
def PlotXYvT(bo,tr,ma):
    '''
    Plot bout bo of trajectory tr in maze ma
    separate curves of x and y vs time
    '''
    ce = tr.ce[bo]
    ce1 = ce[ce>=0] # eliminate the bad frames where ce=-1
    plot([ma.xc[ce1],ma.yc[ce1]],fmts=['r-','g-'],linewidth=1,
         figsize=(8,2),legend=['x','y']);
    
def PlotAllXYvT(filename,title,tr,ma):
    nf = max([len(c) for c in tr.ce]) # number of frames in longest bout
    nb = len(tr.ce) # number of bouts
    fig, axes = plt.subplots(nb, 1, figsize=(8, nb))
    for bo in range(nb):
        ce = tr.ce[bo]
        ce1 = ce[ce>=0] # eliminate the bad frames where ce=-1
        plot([ma.xc[ce1],ma.yc[ce1]],fmts=['r-','g-'],linewidth=1,xlim=[0,nf],
            xhide=True,yhide=True,
            axes=axes[bo])
        axes[bo].text(0,0,'{:d}'.format(bo)) 
    plt.suptitle(title+'. Time axis: {:d} frames'.format(nf), fontsize=9, y=1.0,
                va='bottom')
    plt.savefig('outdata/'+filename+'.pdf', bbox_inches = "tight")
    plt.close()  

def FixTraj(ng,tr,ma):
    '''
    Superseded by FixTraj2()
    Repairs trajectory tr in place. Works only on the discrete positions.
    ng = minimum number of good frames to serve as anchor (a parameter)
    tr = trajectory
    ma = maze
    Returns an array containing the fraction of frames with errors in each bout
    '''
    FE = np.zeros(len(tr.ce)) # fraction of error frames in each bout
    for bo,ce in enumerate(tr.ce): # one bout at a time
        Er = np.zeros(len(ce)) # array that flags errors
        Er[np.where(ce==-1)] = 1 # flag frames where the original had NaN
        for i in range(len(Er)-1):
            if (ce[i]!=-1 and ce[i+1]!=-1 and ma.di[ce[i],ce[i+1]]>1): # flag frames with step size >1
                Er[i]=1; Er[i+1]=1
        FE[bo] = np.sum(Er)/len(Er)
        # print ('bout {:d}: fraction errors = {:.3}'.format(bo, np.sum(Er)/len(Er)))
        if np.sum(Er)>0: # don't do anything if there are no errors
            # flag as bad all the islands of good frames that are <ng long
            Ba = np.where(Er==1)[0] # index array for bad frames
            if Ba[0]<ng: # at start of bout
                Er[:Ba[0]]=1
            if Ba[-1]>len(Er)-ng: # at end of bout
                Er[Ba[-1]+1:]=1
            for i in range(1,len(Ba)): # and in between
                if Ba[i]-Ba[i-1]-1<ng:
                    Er[Ba[i-1]+1:Ba[i]]=1
            Cc = np.copy(ce) # corrected cell positions
            il = [] # the good frame to the left of an interval to be stitched 
            ir = [] # the good frame to the right of an interval to be stitched
            if Er[0]: # special condition if the first frame is bad
                Cc[0] = 0 # set first frame to entry cell
                il += [0] # use that cell as left anchor
            for i in range(len(Er)-1):
                d = Er[i+1] - Er[i]
                if d == 1: # if bad frames start at i+1
                    il += [i] # make frame i a left anchor
                elif d == -1: # if good frames start at i+1
                    ir += [i+1] # make frame i+1 a right anchor
            if Er[-1]: # special condition if the last frame is bad
                Cc[-1] = 0 # set last frame to entry cell
                ir += [len(Er)-1] # use that cell as right anchor 
            # now we have two lists il[] and ir[] that define the anchors left and right of the bad intervals
            for j in range(len(il)): # stitch each of these intervals with the shortest path between the anchor cells
                cl=Cc[il[j]] # left anchor cell
                cr=Cc[ir[j]] # right anchor cell
                Pa = ConnectPath(cl,cr,ma) # shortest path from cl to cr including both
                st = (len(Pa)-1)/(ir[j]-il[j]) # stretch factor to fit that path onto the bad interval
                if st>1:
                    st = 'bout {:d}, frame {:d}: Can\'t stitch, path too long!'.format(bo,il[j])
                    print('\033[0;31;48m'+st+'\033[00m') # makes red text via escape codes
                else:
                    for k in range(il[j]+1,ir[j]):
                        Cc[k]=Pa[int(st*(k-il[j]))] # stretch the path appropriately into the target interval
            tr.ce[bo] = Cc # replace the old with the corrected trajectory for this bout
    return FE

def FixTraj2(tr,ma):
    '''
    Repairs trajectory tr in place. Works only on the discrete positions.
    ng = minimum number of good frames to serve as anchor (a parameter)
    tr = trajectory
    ma = maze
    Returns an array containing the fraction of frames with errors in each bout
    '''
    FE = np.zeros(len(tr.ce)) # fraction of error frames in each bout
    for bo,ce in enumerate(tr.ce): # one bout at a time
        er = np.zeros(len(ce)) # array that flags errors
        er[np.where(ce==-1)]=1 # flag frames where the original had NaN
        n=len(ce)-1 # index of last cell
        cc = np.copy(ce) # corrected cell positions

        # check all adjacent good frames, mark as bad if they're too far apart
        for i in range(n):
            j=i+1
            if (cc[i]!=-1 and cc[j]!=-1 and ma.di[cc[i],cc[j]]>1): # flag frames with step size >1
                er[i]=1
                er[j]=1

        if cc[0]==-1: # first frame special, if bad set to zero, mark as good
            cc[0]=0
            er[0]=0
        if cc[n]==-1: # last frame special, if bad set to zero, mark as good
            cc[n]=0
            er[n]=0

        # check pairs of frames on either side of a bad stretch, mark as bad if too far apart
        # repeat until no more problems
        quit=False
        while not quit:
            de = er.copy()
            de[1:]-=er[:-1] # de[i]=+1 if i is first bad frame, -1 if i is first good frame
            il=np.where(de==1)[0]-1 # last good frame
            ir=np.where(de==-1)[0] # first good frame
            quit=True
            for i,j in zip(il,ir):
                if ma.di[cc[i],cc[j]]>j-i:
                    quit=False
                    if i>0:
                        er[i]=1
                    if j<n:
                        er[j]=1

        # count bad frames that needed fixing
        FE[bo] = np.sum(er)/len(er)

        # stitch across the bad stretches
        if FE[bo]>0: 
            for i,j in zip(il,ir):
                cp = ConnectPath(cc[i],cc[j],ma)
                st = (len(cp)-1)/(j-i) # stretch factor to fit that path onto the bad interval
                if st>1: # this should not happen anymore
                    st = 'bout {:d}, frame {:d}: Can\'t stitch, path too long!'.format(bo,il[j])
                    print('\033[0;31;48m'+st+'\033[00m') # makes red text via escape codes
                else:
                    for k in range(i+1,j):
                        cc[k]=cp[int(st*(k-i))] # stretch the path appropriately into the target interval
        tr.ce[bo] = cc # replace the old with the corrected trajectory for this bout
    return FE
    
def ListErrors(FE):
    '''
    Lists the fraction of errors recorded in array FE
    '''
    for bo in range(len(FE)):
         print ('bout {:d}: fraction errors = {:.3}'.format(bo, FE[bo]))
            
def InspectBouts(tr,ma):
    '''
    Sequentially plots every bout. Keynode and cell trajectory.
    Enter:
    <return> to go to next bout
    'q' to quit
    'b' to mark this bout as bad
    '<int>' to move to bout number <int>
    Returns the list of bad bouts
    '''
    from IPython.display import clear_output
    bad=[]
    i=0
    while i<len(tr.ce):
        print(i,flush=True)
        PlotTraj(i,tr,ma)
        plt.show()
        a=input("enter: ")
        if a=='b':
            bad+=[i,]
        elif a=='q':
            break
        else:
            try:
                j=int(a)
            except:
                j=-1
            if j in range(len(tr.ce)):
                i=j-1    
        i=i+1
        clear_output(wait=True)
    return bad
    
def InspectXYvT(tr,ma):
    '''
    Sequentially plots every bout. Cell trajectory x, y vs t.
    Enter:
    <return> to go to next bout
    'q' to quit
    'b' to mark this bout as bad
    '<int>' to move to bout number <int>
    Returns the list of bad bouts
    '''
    from IPython.display import clear_output
    bad=[]
    i=0
    while i<len(tr.ce):
        print(i,flush=True)
        PlotXYvT(i,tr,ma)
        plt.show()
        a=input("enter: ")
        if a=='b':
            bad+=[i,]
        elif a=='q':
            break
        else:
            try:
                j=int(a)
            except:
                j=-1
            if j in range(len(tr.ce)):
                i=j-1    
        i=i+1
        clear_output(wait=True)
    return bad

def ParseNodeTrajectory(tr,ma):
    '''
    Takes a cell vs time trajectory and returns a node vs time trajectory
    Also adds that trajectory to the tr structure
    ma: maze structure
    tr: trajectory
    '''
    exitstate = len(ma.ru) # exit state = 1 + highest node number
    no = [r[-1] for r in ma.ru] # set of all nodes identified by cell number
    # make the node trajectory as a list of 2D arrays of (node,time), one for each bout. 
    nt = []
    for i,b in enumerate(tr.ce): # for every bout in the cell trajectory
        exittime = tr.fr[i,1]-tr.fr[i,0] # exit time in frames within the bout
        nb=np.array([[exitstate,exittime]]) # bout will end with this exit state
        ni = np.where(np.isin(b,no))[0] # times of all nodes in frames within the bout
        if len(ni)>0: # if no nodes skip this
            nv = ma.rc[b[ni]] # values of those nodes labeled by run number
            vd = np.copy(nv) # value difference from preceding node; we are looking for a change of node
            vd[1:] -= nv[:-1] # difference between current node and previous one
            vd[0] = 1 # first node counts as different
            di = np.where(vd != 0)[0] # index of different nodes
            td = ni[di] # times of the node changes from bout onset
            nd = nv[di] # values of the node after change
            nb = np.append(np.array([nd,td]).T,nb,axis=0) # add the exit state at the end
        nt += [nb] # 2D array of (node,time) for each bout
    tr.no = nt
    return nt

def SmoothTrajectory(tr):
    # remove 1-frame flickers from the node trajectory
    nt = tr.no
    print('before: {} states'.format(sum([len(n) for n in nt])))
    nu=[]
    for j,b in enumerate(nt):
        # on the first round erase all the single-frame states
        erase=[]
        for i in range(1,len(b)-2): # need at least 3 states in the bout excluding the exit state
            if b[i+1,1]-b[i,1]==1 and b[i-1,0]==b[i+1,0]:
                erase+=[i]
        b=np.delete(b,erase,axis=0)
        # on the second round erase all the duplicated states that resulted from this
        erase=[]
        for i in range(1,len(b)-1): # from second state to exit state
            if b[i,0]==b[i-1,0]:
                erase+=[i]
        nu+=[np.delete(b,erase,axis=0)]
    print('after: {} states'.format(sum(len(n) for n in nu)))
    tr.no = nu
    return nu

def PlotCellOccupancy(tr,ma):
    '''
    Plots the number of frames spent in each cell on a log scale
    Returns the occupancy array.
    '''
    nc = len(ma.xc) # number of cells
    fc = sum(np.bincount(b,minlength=nc) for b in tr.ce) # frames spent in each cell
    f=np.log(1+fc) # log function of occupancy
    f=(f-np.min(f))/(np.max(f)-np.min(f)) # map desired function into [0,1]
    PlotMazeFunction(f,ma,mode='cells',numcol=None,figsize=5)
    return fc

def PlotNodeOccupancy(tr,ma):
    '''
    Plots the number of frames spent at each node on a log scale
    Returns the occupancy array.
    '''
    no = NodeOccupancy(tr,ma)
    f = np.log(1+no)
    f = (f-np.min(f))/(np.max(f)-np.min(f))
    PlotMazeFunction(f,ma,mode='nodes',numcol='cyan',figsize=6)
    return no
    
def NodeOccupancy(tr,ma):
    '''
    Returns the number of frames spent at each node.
    '''
    no = np.zeros(len(ma.ru))
    for b in tr.no:
        for n in range(len(b)-1): # all states before the exit state
            no[b[n,0]] += b[n+1,1]-b[n,1] # number of frames spent at this node
    return no
    
def HistoNodeDurations(tr,ma):
    '''
    Histograms the time spent at nodes, grouped by level
    '''
    nd=[[] for n in ma.ru] # an empty list for every node
    for b in tr.no:
        d=b.copy()
        d[:-1,1]=d[1:,1]-d[:-1,1] # if len(d)==1 this should do nothing
        for n in d[:-1]: # exclude exit state
            nd[n[0]] += [n[1]] # number of frames spent at this node 
    ndlev=[[x for n in nd[2**l-1:2**(l+1)-1] for x in n] for l in range(ma.le+1)] # pool by node level
    fig, axes = plt.subplots(4, 2, figsize=(10, 8))
    for i,l in enumerate(ndlev):
        axes[i//2,i%2].hist(l,bins=np.linspace(-0.5,100.5,102));
            
def TallyStepTypes(tr,ma):
    '''
    Counts the steps of each type in the trajectory tr and returns that in a dictionary
    L=0, R=1, l=2, r=3, illegal=-1
    '''
    ta={0:0,1:0,2:0,3:0,-1:0}
    for b in tr.no:
        for i in range(len(b)-1): # no steps from the exit state
            ta[StepType(b[i,0],b[i+1,0],ma)]+=1
    return ta
    
def TallyNodeStepTypes(tr,ma):
    '''
    Counts the steps of each type separately for each node
    Returns a (nodes,4) array
    1st index = node. 2nd index = L=0, R=1, l=2, r=3
    No illegal steps allowed
    '''
    tu = np.zeros((len(ma.ru),4),dtype=int) # 1st index = node. 2nd index = L=0, R=1, l=2, r=3
    for b in tr.no: # bout b
        for i in range(len(b)-1): # no steps from the exit state
            tu[b[i,0],StepType(b[i,0],b[i+1,0],ma)]+=1 
    return tu

def PlotNodeBias(tr,ma):
    '''
    Computes the bias for various steps for each node
    Plots that vs node number for all but the end nodes
    Returns two arrays with the bias for every node:
    bo = out bias = (outleft+outright)/(outleft+outright+inleft+inright)
    bl = left bias = inleft/(inleft+inright)
    '''
    tu = TallyNodeStepTypes(tr,ma)
    n = 2**ma.le-1 # number of nodes below end node level
    bo = (tu[:n,2]+tu[:n,3])/np.sum(tu[:n,:],axis=1) # (outleft+outright)/(outleft+outright+inleft+inright)
    so = np.sqrt((tu[:n,2]+tu[:n,3])*(tu[:n,0]+tu[:n,1])/np.sum(tu[:n,:],axis=1)**3) # std dev
    plot(bo,fmts=['g-'],legend=['back'],linewidth=2,
         xlabel='Node',ylabel='back(back+left+right)',figsize=(10,3),grid=True);
    plt.errorbar(range(len(bo)),bo,yerr=so,fmt='none');  
    plt.show()
    bl = tu[:n,0]/(tu[:n,0]+tu[:n,1]) # inleft/(inleft+inright)
    sl = np.sqrt(tu[:n,0]*tu[:n,1]/(tu[:n,0]+tu[:n,1])**3) # std dev
    plot(bl,fmts=['r-'],legend=['left'],linewidth=2,
         xlabel='Node',ylabel='left/(left+right)',figsize=(10,3),grid=True);
    plt.errorbar(range(len(bl)),bl,yerr=sl,fmt='none'); 
    return bo,bl

def PlotNodeBiasLocation(tr,ma): 
    '''
    Plots left-right turn bias for all nodes below the endnode level
    f: array with left bias for all nodes below the endnode level
    m: maze
    '''
    tu = TallyNodeStepTypes(tr,ma)
    n = 2**ma.le-1 # number of nodes below end node level
    bl = tu[:n,0]/(tu[:n,0]+tu[:n,1]) # inleft/(inleft+inright)
    # extend the bias function to all nodes, make it 0.5 for the endnodes
    f=np.append(bl,np.full(len(ma.ru)-len(bl),0.5))
    # color scale for interpolation: a value of col[j,0] gets mapped into rgb tuple col[j,1:4]
    col=np.array([[0,0,1,0],[0.5,1,1,1],[1,1,0,0]]) 
    PlotMazeFunction(f,ma,mode='nodes',numcol='blue',figsize=6,col=col)

def TallyStrings(tr,m=5):
    '''
    Produces m dictionaries that give the number of occurrences for each j-string 
    up to j=m in the trajectory tr.
    The strings of different length are aligned to all share the first element.
    Returns: a list of dictionaries of type s[string tuple]=number.
    '''
    se=[{} for i in range(m)] # se[j] tallies (j+1)-strings
    for b in tr.no: # bout b
        for i in range(0,len(b)-m+1): # first position of the string
            for j in range(m): # j+1 = length of string
                s=tuple(b[i:i+j+1,0]) # j+1-string starting at i
                if s in se[j]: # tally into the j-th dictionary
                    se[j][s]+=1
                else:
                    se[j][s]=1
    return se

def TallyTwoSteps(tr,ma):
    '''
    Classifies 3-strings of nodes according to the types of the two steps
    'L'=in left, 'R'=in right, 'l'=out left, 'r'=out right.
    Counts and lists the occurrences.
    Also computes an expectation from the average step probabilities from each state,
    i.e. a first-order Markov chain.
    Lists both observed and predicted numbers.
    '''
    ts3 = TallyStrings(tr,3) # number of all possible 3-strings,2-strings,1-strings aligned left
    ts2 = TallyStrings(tr,2) # number of all possible 2-strings,1-strings aligned left
    opn=np.array([list(s)+[ts3[2][s]]+[ts3[1][s[:-1]]*ts2[1][s[1:]]/ts2[0][s[1:-1]]] for s in ts3[2]])
    # For every 3-string the observed and the predicted number
    st=np.zeros((4,4,2)) # [first step type x second step type x observed/predicted]
    for s in opn:
        t1 = StepType(s[0],s[1],ma); t2 = StepType(s[1],s[2],ma) # the 2 steps in this 3-string
        if t1>=0 and t2>=0: # eliminate potential illegal steps from this tally
            st[t1,t2,:]+=s[3:] # accumulate observed and predicted numbers for this step type pair
    lab=['L','R','l','r'] # step type labels
    di={}
    for i in range(4):
        for j in range(4):
            la=lab[i]+lab[j]
            if la != 'Lr' and la != 'Rl': # avoid impossible sequences
                di[la]=st[i,j,:]
    di['in-alt']=di['LR']+di['RL']
    di['in-same']=di['LL']+di['RR']
    di['in-back']=di['Ll']+di['Rr']
    di['out-out']=di['ll']+di['lr']+di['rl']+di['rr']
    di['out-in']=di['lR']+di['rL']
    di['out-back']=di['lL']+di['rR']
    di['in-left']=di['LL']+di['RL']
    di['in-right']=di['RR']+di['LR']
    di['out-left']=di['lR']+di['rl']
    di['out-right']=di['rL']+di['lr']
    print('              obs   pred   obs/pred')
    for t in di:
        print('{:9s}   {:5.0f}  {:5.0f}  {:5.2f}'.format(t,di[t][0],di[t][1],di[t][0]/di[t][1]))

def FindHomeRunNodes(tr,ma):
    '''
    Returns a list of nodes from which the home run starts
    Returns nx4 array containing bout, state, node distance to exit, frame
    '''
    hr=[]
    le=(np.log(range(1,2**(ma.le+1)))/np.log(2)).astype(int) # level for every node
    for i,b in enumerate(tr.no):
        if len(b)>1: # ignore bouts with only an exit state
            d=le[b[:-1,0]] # node level for every state before exit
            if len(d)==1: # only one state, namely 0, no need to search for path
                hr += [i,0,1,tr.fr[i,1]]
            else: # other bouts have at least 3 states
                hr += [b[np.where(np.logical_and(d[1:-1]>d[2:],d[1:-1]>d[:-2]))[0][-1]+1][0]] # last node where diff is positive
        else:
            hr += [np.NaN]
    return hr

def SimulateRandomWalk(t,m,n=1000,r=1):
    '''
    Simulates a node trajectory on the maze ma. Random walk of minimal length n 
    using the measured transition probabilities for `L`, `R`, and `o` from the 
    trajectory tr.
    Trajectory includes exits from the maze, so has somewhat undetermined length.
    t: trajectory from where to measure the probabilities
    m: maze
    n: minimal length of the simulation
    r: random seed
    '''
    np.random.seed(r)
    # make array of states connected to each state, in order `L`, 'R`, `o`
    sta=np.full((len(m.ru),3),-1)
    for i in range(len(m.ru)):
        if StepType(i,m.ch[i][0],m)==0: # child 0 is `L`
            sta[i,0]=m.ch[i][0]
            sta[i,1]=m.ch[i][1]
        else:                          # child 0 is `R`
            sta[i,0]=m.ch[i][1]
            sta[i,1]=m.ch[i][0]
        sta[i,2]=m.pa[i] # parent run
    sta[0,2]=len(m.ru)
    # make array of transition probabilities, in order `L`, 'R`, `o` 
    ta=TallyStepTypes(t,m) # all step types
    tr1=np.array([ta[0],ta[1],ta[2]+ta[3]]) # the default bias for all nodes below endnodes
    tr1=tr1/np.sum(tr1)
    tra=np.zeros((len(m.ru),3))
    tra[0:2**m.le-1,:]=tr1 # same bias for all nodes below endnodes
    tra[2**m.le-1:]=[0,0,1] # can only step `o` from the endnodes
    # simulate the trajectory
    no=[]
    tot=0
    while tot<n:
        s=0 # start in state 0
        bo=[s] # accumulate this bout
        while s != 127: # check for exit
            s=np.random.choice(sta[s].tolist(),p=tra[s]) # random step according to transition probabilities from last state
            bo+=[s] # add state to this bout
        no+=[np.array([bo,np.arange(len(bo))]).T] # add bout to this trajectory along with frames
        tot+=len(bo)
    f = np.array([0]+[len(b) for b in no]) # length of the bouts   
    f = np.cumsum(f)
    fr = np.array([f[:-1],f[1:]]).T    
    return Traj(fr=fr,ce=None,ke=None,no=no,re=None)
    
def FirstTransProb(t,m):
    '''
    Computes 1st order transition probabilities among nodes of maze m in trajectory t
    tra[i,j] is the prob of transition from state i to sta[i,j]
    '''
    sta=TransMatrix(m) # array of nodes connected to each node, in order parent, left child, right child
    # make array of transition probabilities based on current and preceding state 
    se = TallyStrings(t,2) # all occurrences of strings up to length 2
    tra=np.array([[se[1][(i,j)]/se[0][(i,)] if (i,j) in se[1] else 0 for j in s] for i,s in enumerate(sta)]) # trans probs
    return sta,tra
    
def SimulateFirstMarkov(sta=None,tra=None,tr=None,ma=None,n=1000,rs=1):
    '''
    Simulates a node trajectory on the maze ma. First-order markov chain of minimal length n 
    using measured state transition probabilities from the trajectory tr.
    Trajectory includes exits from the maze, so has somewhat undetermined length.
    tr: trajectory from where to measure the probabilities
    ma: maze
    n: minimal length of the simulation
    rs: random seed
    '''
    if sta==None:
        sta,tra=FirstTransProb(tr,ma)
    np.random.seed(rs)
    exit = sta[0,0] # parent state of node 0
    no=[]
    tot=0
    while tot<n:
        s=0 # start in state 0
        bo=[s] # accumulate this bout
        while s != exit: # check for exit
            s=np.random.choice(sta[s].tolist(),p=tra[s]) # random step according to transition probabilities from last state
            bo+=[s] # add state to this bout
        no+=[np.array([bo,np.arange(len(bo))]).T] # add bout to this trajectory along with frames
        tot+=len(bo)
    f = np.array([0]+[len(b) for b in no]) # length of the bouts   
    f = np.cumsum(f)
    fr = np.array([f[:-1],f[1:]]).T    
    return Traj(fr=fr,ce=None,ke=None,no=no,re=None)

def TransMatrix(m):
    '''
    Returns array sta with the connection matrix among nodes in maze m.
    sta[i,:] the 3 nodes connected to node i in order parent, left child, right child
    '''
    # make array of nodes connected to each node, in order `o`, `L`, 'R`
    sta=np.full((len(m.ru),3),-1)
    for i in range(len(m.ru)):
        sta[i,0]=m.pa[i] # parent node
        if StepType(i,m.ch[i][0],m)==0: # child 0 is `L`
            sta[i,1]=m.ch[i][0]
            sta[i,2]=m.ch[i][1]
        else:                          # child 0 is `R`
            sta[i,1]=m.ch[i][1]
            sta[i,2]=m.ch[i][0]
    sta[0,0]=len(m.ru) # special 'exit' node from node 0
    return sta

def SecondTransProb(t,m):
    '''
    Computes 2nd order transition probabilities among nodes of maze m in trajectory t
    Returns arrays sta(n,3) and trb(n,3,3) with n=# of nodes
    sta[i,:] the 3 nodes connected to node i in order o,L,R; i.e. parent, left child, right child
    trb[i,j,k] is the prob of transition from state i to sta[i,k] given prior state was sta[i,j]
    '''
    sta=TransMatrix(m) # array of nodes connected to each node, in order parent, left child, right child
    # make array of transition probabilities based on current and preceding state 
    ta = TallyStrings(t,3) # all occurrences of strings up to length 3
    trb=np.zeros((len(m.ru),3,3)) # 3D array containing transition probability depending on 2-string
    for i in range(len(m.ru)): # i is current state
        for j,sj in enumerate(sta[i]): # sta[i,j] is preceding state
            for k,sk in enumerate(sta[i]): # sta[i,k] is next state
                if (sj,i,sk) in ta[2]:
                    trb[i,j,k]=ta[2][(sj,i,sk)]/ta[1][(sj,i)]
    return sta,trb
    
def SimulateSecondMarkov(sta=None,trb=None,tr=None,ma=None,n=1000,rs=1):
    '''
    Simulates a node trajectory as a second-order markov chain of length n.
    You either provide trajectory tr and maze ma to measure the transition probabilities from the data.
    Or you provide sta and trb as precomputed transition probabilities.
    Trajectory includes exits from the maze, so has somewhat undetermined length.
    n: minimal length of the simulation
    rs: random seed
    '''
    if sta is None:
        sta,trb=SecondTransProb(tr,ma)
    tp=np.full((len(sta)+1,len(sta)+1,3),-1,dtype=float) # helper array, tp[i,j,k] = trans prob from i to sta[i,k] given prev state was j
    for i,st in enumerate(sta):
        for j,sj in enumerate(st):
            for k,sk in enumerate(st):
                tp[i,sj,k]=trb[i,j,k]
    tp0 = (trb[0,1,:]+trb[0,2,:])/2 # 1st order transition prob from state 0, assuming states 1 and 2 are equally likely as preceding
    exit = sta[0,0] # parent state of node 0
    np.random.seed(rs)    
    no=[]
    tot=0
    while tot<n:
        s=0 # start in state 0
        bo=[s] # accumulate this bout
        s=np.random.choice(sta[0].tolist(),p=tp0) # 1st order transition from state 0
        bo+=[s]
        while s != exit: # check for exit
            s=np.random.choice(sta[s].tolist(),p=tp[bo[-1],bo[-2],:]) # random step according to transition probabilities from last state
            bo+=[s] # add state to this bout
        no+=[np.array([bo,np.arange(len(bo))]).T] # add bout to this trajectory along with frames
        tot+=len(bo)
    f = np.array([0]+[len(b) for b in no]) # length of the bouts   
    f = np.cumsum(f)
    fr = np.array([f[:-1],f[1:]]).T    
    return Traj(fr=fr,ce=None,ke=None,no=no,re=None)

def Simulate2ndMarkovBias(tr,ma,n=1000,rs=1):
    '''
    Computes avg and std of the 6-value bias array over all nodes in a level.
    Pattern:
        Bf  Bl/a
        Lf  Lo
        Rf  Ro
    If alt==False, the [0,1] value (printed in red) is Bl (fraction of left turns)
    If alt==True, the [0,1] value (printed in blue) is Ba (fraction of alternating turns)
    '''
    sta,trb=SecondTransProb(tr,ma)
    bi=np.array([Bias(i,ma,trb,alt=True) for i in range(1,2**ma.le-1)]) # all nodes from level 1 to le-1
    ba=np.average(bi,axis=0) # average across all
    ba[1:3,0]=(ba[1,0]+ba[2,0])/2 # average Lf and Rf
    ba[1:3,1]=(ba[1,1]+ba[2,1])/2 # average Lo and Ro
    trc=np.copy(trb)
    for i in range(2**ma.le-1): # all nodes below endnodes
        tr=trc[i] # translate the bias array to the trans prob array
        tr[0,0]=1-ba[0,0];tr[0,1]=ba[0,0]*ba[0,1];tr[0,2]=ba[0,0]*(1-ba[0,1])
        tr[1,1]=1-ba[1,0];tr[1,0]=ba[1,0]*ba[1,1];tr[1,2]=ba[1,0]*(1-ba[1,1])
        tr[2,2]=1-ba[2,0];tr[2,0]=ba[2,0]*ba[2,1];tr[2,1]=ba[2,0]*(1-ba[2,1])
        if StepType(ma.pa[i],i,ma)==0: # an L node, so 'alt' refers to R turns
            tr01=tr[0,1] # swap the left and right turn biases coming from the stem of the T 
            tr[0,1]=tr[0,2]
            tr[0,2]=tr01
    trc[0,0,:]=0 # no forward steps from exit
    return SimulateSecondMarkov(sta=sta,trb=trc,tr=None,ma=None,n=n,rs=rs)
    
def Bias(i,m,trb,alt=False):
    '''
    Computes 6 biases for node i of maze m based on transition probs in trb
    Can enter the T junction from bottom (B), left (L), or right (R)
    From each direction, can either go forward (f) or reverse back to the preceding state.
    If go forward from L or R, can go out (o) or into the maze
    If go forward from B, can go left (l) or right.
    Alternatively, if alt==True score this as an alternating (a) vs same direction turn.
    So the components of the bias are: Bf, Bl, Lf, Lo, Rf, Ro
    '''
    def Norm(x):
        if x[0]==0:
            return 0
        else:
            return x[0]/sum(x)
    tr=trb[i]
    Bf=Norm([tr[0,1]+tr[0,2],tr[0,0]]) # Bf = forward bias from B
    if alt and StepType(m.pa[i],i,m)==0: # an L node, so 'alt' refers to right turns
        Bl=Norm([tr[0,2],tr[0,1]])
    else:
        Bl=Norm([tr[0,1],tr[0,2]]) # Bl = left bias when stepping forward from B
    Lf=Norm([tr[1,0]+tr[1,2],tr[1,1]]) # Lf = forward bias from L
    Lo=Norm([tr[1,0],tr[1,2]]) # Lo = outward bias if forward from L
    Rf=Norm([tr[2,0]+tr[2,1],tr[2,2]]) # Rf = forward bias from R
    Ro=Norm([tr[2,0],tr[2,1]]) # Ro = outward bias if forward from R
    return np.array([[Bf, Bl],[Lf, Lo],[Rf, Ro]])

def ListAvgNodeBias(tr,ma,alt=False):
    '''
    Computes avg and std of the 6-value bias array over all nodes in a level.
    Pattern:
        Bf  Bl/a
        Lf  Lo
        Rf  Ro
    If alt==False, the [0,1] value (printed in red) is Bl (fraction of left turns)
    If alt==True, the [0,1] value (printed in blue) is Ba (fraction of alternating turns)
    '''

    class color:
       PURPLE = '\033[95m'
       CYAN = '\033[96m'
       DARKCYAN = '\033[36m'
       BLUE = '\033[94m'
       GREEN = '\033[92m'
       YELLOW = '\033[93m'
       RED = '\033[91m'
       BOLD = '\033[1m'
       UNDERLINE = '\033[4m'
       END = '\033[0m'
   
    sta,trb=SecondTransProb(tr,ma)
    bi=np.array([Bias(i,ma,trb,alt) for i in range(2**ma.le-1)]) # compute 6-parameter bias for each node
    ba=[]; bs=[]
    for l in range(ma.le): # for each level
        ba+=[np.average(bi[2**l-1:2**(l+1)-1,:,:],axis=0)] # average bias over nodes in that level
        bs+=[np.std(bi[2**l-1:2**(l+1)-1,:,:],axis=0)] # std dev of bias over nodes in that level

    col=[[color.END,color.RED],[color.END,color.END],[color.END,color.END]]
    if alt:
        col[0][1]=color.BLUE
    for l in range(ma.le):
        for i in range(3):
            for j in range(2):
                print((col[i][j]+'{:.2f} ± {:.2f}  '+color.END).
                      format(ba[l][i,j],bs[l][i,j]),end='')
            print()
        print()

def StringEntropy(tr,ma,n=50,mode='SA',endnodes=True):
    '''
    Computes the entropy for strings of length 1,2,...,n in the node trajectory tr.no
    through maze ma
    modes:
    'SS': all elements of the string are states
    'SA': all elements are states except the last one is the action leading to 
          the last state, which can be {In Left, In Right, or Out}
    'AA': all elements are actions among {In Left, In Right, Out Left, Out Right}
          except the last action, which can be {In Left, In Right, or Out} 
    endnodes: include actions taken at endnodes?
    Returns:
    hs[i] = conditional entropy of the last element given the i preceding ones
    hsa[i] = joint entropy of the last element with the i preceding ones
    hsb[i] = joint entropy of the i+1 elements preceding the last one
    num = number of strings inspected
    '''
    if mode=='AA':
        n1=n-1 # in this case return only n-1 values for the n-1 actions in an n-string
    else:
        n1=n
    hsa=np.zeros(n1) # entropy for substrings of length 1,...,n1
    hsb=np.zeros(n1-1) # entropy for history substrings of length 1,...,n1-1 
        
    # compute tally of n-strings from the data
    sf={}
    num=0 # number of states inspected
    nt=tr.no
    for b in nt:
        for i in range(len(b)-n+1): # step along the bout, sliding window of n states, include exit state
            c = b[i:i+n,0].copy() # window of n states, forget about time for now
            if not (endnodes==False and c[-2]>2**ma.le-2): # eliminate if final step from an endnode
                num+=1 # accumulate number of states inspected
                if mode=='SA':
                    c[-1]= StepType2(c[-2],c[-1],ma) # convert the last state into type 2 action
                elif mode=='AA':
                    c[-1]= StepType2(c[-2],c[-1],ma) # convert the last state into type 2 action
                    c[1:-1]= [StepType(c[i-1],c[i],ma) for i in range(1,n-1)] # convert the others into type 1 action
                    c=c[1:] # strip the first element because there are only n-1 actions
                s = tuple(c)
                if s in sf: # and tally the different outcomes
                    sf[s]+=1
                else:
                    sf[s]=1

    # compute entropy of the full n1-string
    sfm=np.array([sf[s] for s in sf])
    hsa[n1-1]=np.sum(-(sfm/num)*np.log(sfm/num))/np.log(2)
    
    # compute entropies of shorter strings ending in the last state
    sf1=sf.copy() # keep a copy of the full tally for later
    sf2={}
    for m in range(n1-1,0,-1): # m = length of the string to be analyzed
        # compute tally of m-strings from that of (m+1)-strings
        temp = sf2 # point to the old dictionary that can now be overwritten
        sf2 = sf1 # point to the preceding dictionary that we need
        sf1 = temp # point to the old dictionary
        sf1.clear() # erase the old dictionary, frees memory
        for s2 in sf2:
            s1=s2[1:] # cut first element from string, this is the most distant state
            if s1 in sf1: # and tally the different outcomes of the remaining string
                sf1[s1]+=sf2[s2]
            else:
                sf1[s1]=sf2[s2]
        # compute entropy of the m-strings
        sfm=np.array([sf1[s1] for s1 in sf1])
        hsa[m-1]=np.sum(-(sfm/num)*np.log(sfm/num))/np.log(2)
#     print('Current Actions:',sf1) # debugging
    
    # compute entropies of history strings ending in the second-to-last state
    sf1=sf # Won't need the full tally after this, so can overwrite it
    sf2={}
    for m in range(n1-1,0,-1): # m = length of the string to be analyzed
        # compute tally of m-strings from that of (m+1)-strings
        temp = sf2 # point to the old dictionary that can now be overwritten
        sf2 = sf1 # point to the preceding dictionary that we need
        sf1 = temp # point to the old dictionary
        sf1.clear() # erase the old dictionary, frees memory
        if m==n1-1: # for first tally of history, cut the last state from the strings
            for s2 in sf2:
                s1=s2[:-1] # cut last element from string
                if s1 in sf1: # and tally the different outcomes
                    sf1[s1]+=sf2[s2]
                else:
                    sf1[s1]=sf2[s2]
        else: # for shorter histories cut the first state
            for s2 in sf2:
                s1=s2[1:] # cut first element from string
                if s1 in sf1: # and tally the different outcomes
                    sf1[s1]+=sf2[s2]
                else:
                    sf1[s1]=sf2[s2]
        # compute entropy of the m-strings of history
        sfm=np.array([sf1[s1] for s1 in sf1])
        hsb[m-1]=np.sum(-(sfm/num)*np.log(sfm/num))/np.log(2)
#     print('History Actions:',sf1) # debugging

    # compute conditional entropies by subtracting entropy of history string
    hs=hsa.copy() # keep the joint entropy to report it
    hs[1:]-=hsb
    
    return hs, hsa, hsb, num # cond entr, joint entr, history entr, num strings

def PlotStringEntropy(tr,ma,n=50,mode='SS',endnodes=True):
    '''
    Plots the joint entropy (left) and conditional entropy (right) for strings up to length n
    in the trajectory tr through maze ma.
    See StringEntropy() regarding 'mode' and 'endnodes'.
    '''
    hs, hsa, hsb, num = StringEntropy(tr,ma,n,mode,endnodes)
    print('log(number of strings) = {:.2f}'.format(np.log(num)/np.log(2)))
    fig,ax=plt.subplots(1,2,figsize=(8,3))
    if mode=='AA':
        word='actions'
    else:
        word='states'
    plot(np.arange(1,len(hsa)+1),hsa,xlabel='Number of '+word,ylabel='Joint Entropy (bits)',
        ylim=[0,max(hsa)*1.05],axes=ax[0]);
    plot(hs,xlabel='Number of preceding '+word,ylabel='Conditional Entropy (bits)',fmts=['r-'],
        ylim=[0,max(hs)*1.05],axes=ax[1]);    
    return hs, hsa, hsb, num
    
def FindPathsToExit(tr,ma):
    '''
    Finds all monotonic trajectories to the exit in maze ma during trajectory tr (home runs)
    Computes the node distance of those paths
    Bouts that have no node before the exit state are ignored
    Returns nx4 array containing bout, starting node, node distance, absolute frame
    '''
    wd=np.array([ma.di[r[-1],ma.ru[0][0]] for r in ma.ru])+1 # cell distance to exit for every node
    ptn=[]
    hr=np.zeros(len(tr.no),dtype=int)
    for i,b in enumerate(tr.no):
        if len(b)>1: # ignore bouts with just the exit state
            d=wd[b[:-1,0]] # distance to exit excluding exit state
            j = len(d)-1 # start at last state
            k = j-1 # step back in time
            if k>=0:
                while d[k]>d[k+1]: # as long as distance increases
                    k-=1
                    if k==-1: # stop at the start of the bout
                        break
            k+=1 # this is the first state in this monotonic path    
            ptn.append([i, b[k,0], j-k+1, b[k,1]+tr.fr[i,0]]) # bout, starting state, node distance, absolute frame
    return np.array(ptn) # bout, starting state, node distance, absolute frame

def FindPathsToNode(n,tr,ma):
    '''
    Finds all monotonic paths to an arbitrary node n in maze ma during trajectory tr
    Computes the node distance of those paths
    Returns nx4 array containing bout, frame in bout, node distance, absolute frame
    '''
    wd=np.array([ma.di[r[-1],ma.ru[n][-1]] for r in ma.ru]) # cell distance to target for every node
    ptn=[]
    for i,b in enumerate(tr.no):
        if len(b)>1: # ignore bouts with just one state
            d=wd[b[:-1,0]] # distance to target node excluding exit state
            js=np.where(d[1:]==0)[0]+1 # states at the target, ignore first state
            for j in js:
                k = j-1
                while d[k]>d[k+1]:
                    k-=1
                    if k==-1:
                        break
                k+=1 # first state in this path    
                ptn.append([i, k, j-k, b[k,1]+tr.fr[i,0]]) # bout, frame in bout, node distance, absolute frame
    return np.array(ptn,dtype=int) # bout, state in bout, node distance, absolute frame

def PlotPathsToNode(n,tr,ma):
    '''
    Plots all monotonic paths to an arbitrary node n in maze ma during trajectory tr
    Plots the node distance of those paths
    Returns nx4 array containing bout, frame in bout, node distance, absolute frame
    '''
    ptn = FindPathsToNode(n,tr,ma)
    plot(ptn[:,3],ptn[:,2],fmts=['o'],markersize=2,figsize=(8,3),
        xlabel='Time (frames)',ylabel='Node distance to node {}'.format(n),
        xlim=[tr.fr[0,0],tr.fr[-1,-1]],ylim=[-1,13]);
    return ptn # bout, frame in bout, node distance, absolute frame

def PlotPathsToExit(tr,ma):
    '''
    Plots all monotonic trajectories to the exit in maze ma during trajectory tr (home runs)
    Computes the node distance of those paths
    Bouts that have no node before the exit state are ignored
    Returns nx4 array containing bout, frame in bout, node distance, absolute frame
    '''
    ptn = FindPathsToExit(tr,ma)
    plot(ptn[:,3],ptn[:,2],fmts=['o'],markersize=2,figsize=(8,3),
        xlabel='Time (frames)',ylabel='Node distance to exit',
        xlim=[tr.fr[0,0],tr.fr[-1,-1]],ylim=[-1,8]);
    return ptn # bout, frame in bout, node distance, absolute frame
    
def TimeInMaze(f,tf):
    '''
    f = absolute time measured in video frames
    tf = trajectory
    returns the time in maze in seconds
    '''
    n=len(tf.fr)
    ti=0
    i=0
    while tf.fr[i,1]<f:
        ti+=tf.fr[i,1]-tf.fr[i,0]
        i+=1
        if i==n:
            break
    ti+=f-tf.fr[i,0]
    return ti/30
    
def FrameInExpt(t,tf):
    '''
    t = time in maze in seconds
    tf = trajectory
    returns absolute time during expt in frames
    '''
    n=len(tf.fr)
    ti=0
    i=0
    while True:
        tb=(tf.fr[i,1]-tf.fr[i,0])/30 # length of this bout in s
        if ti+tb>t or i==n-1:
            break
        ti+=tb
        i+=1
    return tf.fr[i,0]+(t-ti)*30    

def SplitModeClips(tf,ma,re=True):
    '''
    Splits a trajectory into clips corresponding to the modes leave=0, drink=1, explore=2.
    Each clip records the bout number, start state in the bout, number of steps, and mode.
    re=rewarded? If re=False, only modes leave and explore are recognized
    '''
    water=116 # node number for water port
    ed=np.array([ma.di[r[-1],0] for r in ma.ru])+1 # cell distance to exit for every node
    wd=np.array([ma.di[r[-1],ma.ru[water][0]] for r in ma.ru])+1 # cell distance to water for every node
    leave=0; drink=1; explore=2 # the 3 modes
    cl=[] # list to accumulate clips
    for i,b in enumerate(tf.no): # for each bout
        if len(b)>1: # ignore bouts with just the exit state
            edb=ed[b[:-1,0]] # distance to exit for every state in the bout, ignoring exit state
            wdb=wd[b[:-1,0]] # distance to water for every state in the bout
            # find and reverse the leave path
            j = len(edb)-1 # start at last state before the exit state
            k = j-1 # step back in time
            if k>=0:
                while edb[k]>edb[k+1]: # as long as distance increases
                    k-=1
                    if k==-1: # stop at the start of the bout
                        break
            k+=1 # this is the first state in this monotonic path    
            cl.append([i, k, j-k, leave]) # bout, starting node index within the bout, number of steps, mode
            # find and reverse the subsequent clips
            while k>0: # stop at the start of the bout
                if b[k,0]== water and re==True:
                    j = k # start at last state of preceding clip
                    k = j-1 # step back in time
                    while k>=0 and wdb[k]>wdb[k+1]: # continue until distance to water increases or start of bout
                        k-=1
                    k+=1 # this is the first state in this monotonic path    
                    cl.append([i, k, j-k, drink]) # bout, starting node index within the bout, number of steps, mode
                else:
                    j = k # start at last state of preceding clip
                    k = j-1 # step back in time
                    while k>=0 and b[k,0]!= water: # continue until hit water port or start of bout
                        k-=1
                    if k==-1: # stopped at the start of the bout
                        k+=1
                    cl.append([i, k, j-k, explore]) # bout, starting node index within the bout, number of steps, mode
    cl=np.array(cl) # make into nx4 array
    cl=cl[np.lexsort((cl[:,1],cl[:,0]))] # sort by bout then start node
    return cl

def NewNodesMerge3(tf,ma,le):
    '''
    Computes the number of new nodes encountered in a given window of total nodes,
    a measure of exploration efficiency.
    Concatenates all bouts together. Only counts time in maze.
    Limits itself to nodes of level le.
    Also computes the video frames required on average for the given window of total nodes.
    Returns a 3xn array:
    wcn[0,:]=window width in frames
    wcn[1,:]=window width in nodes
    wcn[2,:]=new nodes per window
    '''
    nb=len(tf.no) # number of bouts
    du=np.sum(tf.fr[:,1]-tf.fr[:,0]) # total duration in frames
    en=list(range(2**le-1,2**(le+1)-1)) # list of node numbers in level le
    ce=np.concatenate([x[:,0] for x in tf.no]) # concatenate all the bouts, only the nodes, not the times
    ei=np.where(np.isin(ce,en))[0] # index of all the desired node states
    if len(ei)>0: # if there is at least one state
        cn=np.copy(ce[ei]) # only the desired nodes
        lc=len(cn) # number of desired nodes encountered
        c=np.array([2,3,6,10,18,32,56,100,180,320,560,1000,1800,3200,5600,10000]) # window width in nodes
        c=c[np.where(c<lc)] # use only those shorter than full length
        c=np.append(c,lc) # add full length as last value
        n=[np.average(np.array([len(set(cn[j:j+c1])) for j in range(0,lc-c1+1,(lc-c1)//(lc//c1)+1)])) for c1 in c]
            # average number of distinct nodes in slightly overlapping windows of size w 
        w=du/lc*c # scale the window widths to frames based on duration of the experiment
    else:
        w=np.array([]); c=np.array([]); n=np.array([])
    wcn=np.array([w,c,n])
    return wcn
    
def NewNodes4(ns,fpn=1):
    '''
    Computes the number of new nodes encountered in a given window of total nodes,
    a measure of exploration efficiency.
    ns=sequence of nodes
    fpn=frames per node to provide time scaling
    '''
    lc=len(ns) # number of nodes in the sequence
    c=np.array([2,3,6,10,18,32,56,100,180,320,560,1000,1800,3200,5600,10000]) # window width in nodes
    c=c[np.where(c<lc)] # use only those windows shorter than full length of the sequence
    c=np.append(c,lc) # add full length as last value
    n=[np.average(np.array([len(set(ns[j:j+c1])) for j in range(0,lc-c1+1,(lc-c1)//(lc//c1)+1)])) for c1 in c]
            # average number of distinct nodes in slightly overlapping windows of size w 
    w=c*fpn # scale the window widths to frames based on duration of the experiment
    return np.array([w,c,n]) # window width, window duration, new nodes per window
    
def MakeRandomWalk(ma,n=1000,rs=1,bi=2/3):
    '''
    Simulates a node trajectory on the maze ma. Random walk of length >n nodes.
    Starts at node #0. One long bout. One cell per frame. Then exit.
    Every node has p=1/3 for the 3 actions, 
    except #0 (p=1/2 for `L` and `R`) and endnodes (p=1 for `o`).
    Trajectory includes exits from the maze, so has somewhat undetermined length.
    ma: maze
    n: minimal length of the simulation in nodes
    rs: random seed
    returns: trajectory
    '''
    np.random.seed(rs)
    # make array of nodes connected to each node, in order `L`, 'R`, `o`
    sta=np.full((len(ma.ru),3),-1)
    for i in range(len(ma.ru)):
        if StepType(i,ma.ch[i][0],ma)==0: # child 0 is `L`; these distinctions might be useful lateron
            sta[i,0]=ma.ch[i][0]
            sta[i,1]=ma.ch[i][1]
        else:                          # child 0 is `R`
            sta[i,0]=ma.ch[i][1]
            sta[i,1]=ma.ch[i][0]
        sta[i,2]=ma.pa[i] # parent run
    sta[0,2]=len(ma.ru) # exit code
    # make array of transition probabilities, in order `L`, 'R`, `o` 
    tra=np.zeros((len(ma.ru),3))
    tra[0]=[1/2,1/2,0] # from #0 can only go L or R
    tra[1:2**ma.le-1]=[bi*0.5,bi*0.5,1-bi] # same bias for all nodes below endnodes
    tra[2**ma.le-1:]=[0,0,1] # can only step `o` from the endnodes
    # make the trajectory in terms of nodes
    ce=[]
    s1=0 # go to node #0
    ce+=ma.ru[s1] # accumulate cells to get to node #0
    tot=0
    while len(ce)<n:
        s0=s1
        s1=np.random.choice(sta[s0].tolist(),p=tra[s0]) # random step according to transition probabilities from last state
        if s1>s0: # stepping in
            ce+=ma.ru[s1]
        else: # stepping out
            ce+=ma.ru[s0][-2::-1] # reverse of run s0 minus the last cell
            ce+=[ma.ru[s1][-1]] # last cell of run s1
    ce+=HomePath(ce[-1],ma)[1:] # path to exit
    fr=np.array([[0,len(ce)]]) # frames assuming speed of one cell per frame
    ce=[np.array(ce)]
    rw=Traj(fr=fr,ce=ce,ke=None,no=None,re=None)
    ParseNodeTrajectory(rw,ma)
    return rw
    
def Make2ndMarkov(ma,n=1000,rs=1,bi=[2/3,1/2,2/3,1/2]):
    '''
    Simulates a node trajectory on the maze ma. 
    Random walk of length >n nodes.
    Multiple bouts.
    No cell list. Frames listed at 1 frame per node.
    ma = maze
    n = number of nodes
    rs = random seed
    bi = array of 4 biases=[Bf, Ba, Lf, Lo]
    Bf = prob of going forward when arriving from bottom of the T
    Ba = prob of making an alternating (instead of repeating) turn when going forward from bottom
    Lf = prob of going forward when arriving from L branch (or R branch)
    Lo = prob of turning out (instead of in) when going forward from L (or R)
    Default values are unbiased random walk with no memory.
    returns: trajectory
    '''
    np.random.seed(rs)
    sta=TransMatrix(ma) # matrix of connections between nodes
    trb=np.zeros((len(ma.ru),3,3)) # 3D array containing transition probability depending on 2-string
    Bf,Ba,Lf,Lo = bi
    for i in range(2**ma.le-1): # all nodes below endnodes
        tr=trb[i] # 3x3 trans prob array for node i
        tr[0,0]=1-Bf;tr[0,1]=Bf*Ba;tr[0,2]=Bf*(1-Ba)
        tr[1,1]=1-Lf;tr[1,0]=Lf*Lo;tr[1,2]=Lf*(1-Lo)
        tr[2,2]=1-Lf;tr[2,0]=Lf*Lo;tr[2,1]=Lf*(1-Lo)
        if StepType(ma.pa[i],i,ma)==0: # an L node, so 'alt' refers to R turns
            tr01=tr[0,1] # swap the left and right turn biases coming from the stem of the T 
            tr[0,1]=tr[0,2]
            tr[0,2]=tr01
    trb[0,0,:]=0 # no forward steps from exit
    for i in range(2**ma.le-1,2**(ma.le+1)-1): # all endnodes
        trb[i]=[[1,0,0],[0,0,0],[0,0,0]] # can only reverse
    return SimulateSecondMarkov(sta=sta,trb=trb,tr=None,ma=None,n=n,rs=rs)
    
def Ln(x):
    '''
    protects log function against zero arguments
    '''
    if isinstance(x,(tuple,list,np.ndarray)):
        return np.array([np.log(y) if y > 0 else np.nan for y in x])
    else:
        if x<=0:
            return np.nan
        else:
            return np.log(x)
            
def xlogx(x):
    '''
    protects x*log(x) against zero arguments
    '''
    if isinstance(x,(tuple,list,np.ndarray)):
        return np.array([y*np.log(y) if y > 0 else 0 for y in x])
    else:
        if x<=0:
            return 0
        else:
            return x*np.log(x)

def Entropy(p):
    '''
    returns the entropy of the distribution p[], normalizes it first
    '''
    p1=p/np.sum(p)
    return -np.sum(xlogx(p1))/np.log(2)

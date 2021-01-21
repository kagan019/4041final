SHOWIMG = True

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

SEED= 342384
#center of coordinate lattice is (0,0)
radius_spawn_ring = 50#500
desired_ptcls = 1000#100000
sticking_prb = 0.7
lattice_sidel = 2*radius_spawn_ring+1

randpool = None
normpool = None
rctr = 0
nctr = 0
def randreset():
    global rctr,nctr,normpool,randpool
    np.random.seed(SEED)
    randpool = np.random.uniform(size=99999999)
    normpool = np.random.normal(scale=2, size=99999999)
    rctr = 0
    nctr = 0

def unif():
    global rctr
    rctr += 1
    if rctr > len(randpool):
        exit("pool size too small")
    return randpool[rctr-1]


def rnorm():
    global nctr
    nctr += 1
    if nctr > len(normpool):
        exit("pool size too small")
    return normpool[nctr-1]
    

def asimg(sidel,pts):
    w,h = sidel,sidel
    #plot a set of points as an image
    im = np.zeros((w,h))
    for (x,y) in pts:
        im[y+h//2][x+w//2] = 100
    plt.matshow(im)
    plt.show()

def circle(r):
    #discrete lattice points that make a circle
    #upper right corner of circle
    quarter = []
    x_ = r
    y_ = 1
    cnes = lambda a,b:abs(a**2+b**2-r**2)
    while x_ > 0:
        while x_ > 0 and (cnes(x_,y_) > cnes(x_-1,y_)
        or cnes(x_-1,y_) < cnes(x_-1,y_+1)):
            x_ -= 1
            quarter += [(x_,y_)]
        quarter += [(x_,y_)]
        y_ += 1
    fullcrcl = []
    for x,y in set(quarter):
        x,y = (int(x),int(y))   
        fullcrcl += [(x,y)]
        fullcrcl += [(-x,-y)]
        fullcrcl += [(-x,y)]
        fullcrcl += [(x,-y)]
    fullcrcl += [(int(r),int(0)),(int(0),int(r)),(int(0),int(-r)),(int(-r),int(0))]
    return list(set(fullcrcl))

def line(sx,sy,ex,ey):
    #Bresenham's algorithm
    fromto = lambda s,e: range(s,e+int(np.sign(e-s)),int(np.sign(e-s)))
    deltax = ex - sx
    deltay = ey - sy
    ret = []
    if deltax == 0:
        if deltay == 0:
            return [(sx,sy)]
        for y in fromto(sy,ey):
            ret += [(sx,y)]
        return ret
    if (abs(deltax) >= abs(deltay)):
        deltaerr = abs(deltay / deltax)
        error = 0.0
        y = sy
        for x in fromto(sx,ex):
            ret += [(x, y)]
            error += deltaerr
            if error >= 0.5:
                y = y + int(np.sign(deltay))
                error -= 1.0
        return ret
    else:
        return [(v,u) for u,v in line(sy,sx,ey,ex)]

randreset()
SHOWIMG = True


#notes on notation
#([], [], []) is a column vector of rows 
# an embeded row [0 0 [0 0] 0 0 ] = a row [0 0 0 0 0 0]

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(342384)
#center of coordinate lattice is (0,0)
radius_spawn_ring = 50
desired_ptcls = 1000
sticking_prb = 0.7
lattice_sidel = 2*radius_spawn_ring+1

randpool = np.random.uniform(size=9999)
rctr = 0
def unif():
    global rctr
    global randpool
    rctr += 1
    if rctr > len(randpool):
        randpool = np.random.uniform(size=9999)
        rctr = 1
    return randpool[rctr-1]

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


def gaussian_rdm_walk():
    print("gaussian_rdm_walk")    
    aggregate = set([(0,0)])
    crcl = circle(radius_spawn_ring)
    #asimg(lattice_sidel,crcl) #is in fact a circle

    def spawn():
        ch = np.random.choice(range(len(crcl)))
        return crcl[ch]
    
    def gauss2():
        return np.random.normal(scale=2, size=2)

    directions = [(0,1),(0,-1),(1,0),(-1,0)]
    ptcl = None
    while len(aggregate) < desired_ptcls:
        if ptcl is None:
            ptcl = spawn()
        dx,dy = gauss2()
        dx,dy = int(round(dx)),int(round(dy))

        x,y = ptcl
        #optimization: the particle moves in larger steps further away
        if x**2+y**2 > (radius_spawn_ring/4)**2:
            dx,dy = 2*dx,2*dy
        tryx,tryy = (x+dx,y+dy)
        # my optimization for escape
        #this makes for a more fair comparison considering diffusion_collapse's 
        #boundary conditions
        if tryx**2+tryy**2 > radius_spawn_ring**2: 
            continue   
        #movement                            
        if (tryx,tryy) not in aggregate:          
            ptcl = (tryx,tryy)                     
        #collision
        for (ax,ay) in directions:
            x,y = ptcl
            test = (x+ax,y+ay)
            if test in aggregate:
                if unif() < sticking_prb:
                    #stick!
                    aggregate.add(ptcl)
                    ptcl = None
                    break
    if SHOWIMG:
        asimg(lattice_sidel,list(aggregate))
    return aggregate

def diffusion_collapse():
    print("diffusion_collapse")
    N=int(radius_spawn_ring*np.pi) #num ptcls to simulate at a time
    diffusion_matrix = np.zeros((lattice_sidel,lattice_sidel))
    aggregate = set()
    def index(cd):
        return (radius_spawn_ring+cd[0],radius_spawn_ring+cd[1])
    def coord(idx):
        return (-radius_spawn_ring+idx[0],-radius_spawn_ring+idx[1])

    crcl = circle(radius_spawn_ring)

    def collapse():
        for i in range(lattice_sidel):
            for j in range(lattice_sidel):
                diffusion_matrix[j][i] = (1.
                    if int(unif()*1000)/1000 < diffusion_matrix[j][i] 
                    else 0.
                ) 
    
    def ADI(steps):
        # let T be the symbol for the diffusion matrix, and
        # is the matrix of discretized space with values as
        # particle probabilities. Say T is MxN=L.

        #remember, x is indexed by i and y by j
        # So, a matrix that looks like
        # |------- +x
        # |  ( [ a u ] , 
        # |    [ b v ] ,
        # +y   [ c w ] )
        # is indexed like T[j][i] (not i j); and
        # the origin of the coordinate system is in the upper left

        # let TX[n] = (T[j][i-1], T[j][i], T[j][i+1]) at timestep n
        # let TY[n] = (T[j-1][i], T[j][i], T[j+1][i]) at timestep n
        # let A1 = [-k (1+2k) -k]
        # let A2 = [k (1-2k) k]
        # where k=Ddeltat/(2*deltax*deltay) for diffusion coefficient D.
        # k will be 1 for our purposes (unitless)

        #first half step:  (A1)TX[n+1/2] = (A2)TY[n]
        #second half step: (A1)TY[n+1] = (A2)TX[n+1/2]

        #this system describes a discretized diffusion equation with good stability properties.
        # We'll impose, as boundary conditions, that values of T outside this matrix = 0, and also
        # that particles do not want to travel outside the matrix. 

        # Let's look at the tridiagonal formulation of the full linear system
        # described by each step. To begin, well define 'row-wrapped'
        # rowT[n] = (T[0][0],T[0][1],T[0][2],...,T[1][0],...,T[M-1][N-1]) at timestep n (l=j*N+i)
        # and 'col-wrapped'
        # colT[n] = (T[0][0],T[1][0],T[2][0],...,T[0][1],...,T[M-1][N-1]) at timestep n (m=i*M+j)
        # the row-wrapped form is a long vector taken from placing the rows of T side-by-side.
        # the col-wrapped form is analogous.
        
        # Next, we define the LxL diagonal matrices,
        # P(Z) = [Z[1] Z[2] 0 0 0...]
        #        [Z 0 0 ...], 
        #        [ 0 Z 0 0 ...], 
        #           ..., 
        #        [0 0 0   ...      Z],
        #        [0 0 0 ... Z[0] Z[1]])
        # Q(Z) = [Z[1] Z[2] 0 0 0...]
        #        [Z 0 0 ...], 
        #        [ 0 Z 0 0 ...], 
        #           ..., 
        #        [0 0 0   ...      Z],
        #        [0 0 0 ... Z[0] Z[1]])
        # the differences are:
        # every P[j*N][j*N-1] = 0 or dne and every P[j*N-1][j*N+1] = 0 or dne
        # every Q[i*M][i*M-1] = 0 or dne and every Q[i*M-1][i*M+1] = 0 or dne
        # from the boundary conditions.

        # first half step: (P(A1))rowT[n+1/2] = (Q(A2))colT[n]
        # second half step: (Q(A1))colT[n+1] = (P(A2))rowT[n+1/2]

        k = 20
        M = lattice_sidel
        N = lattice_sidel
        L = M*N
        A1 = (-k, (1+2*k), -k)
        A2 = (k, (1-2*k), k)
        import scipy.sparse as sp
        import scipy.sparse.linalg
        diags = lambda u: [u[0]*np.ones(L-1),u[1]*np.ones(L),u[2]*np.ones(L-1)]
        def Pdiags(Z):
            Pdiags = diags(Z)
            for j in range(M):
                if j > 0:
                                # -1 adjusted diagonal
                    Pdiags[0][j*N -1] = 0.
                if j < M-1:
                    Pdiags[2][j*N-1] = 0.
            return Pdiags
        def Qdiags(Z):
            Qdiags = diags(Z)
            for i in range(N):
                if i > 0:
                                # -1 adjusted diagonal
                    Qdiags[0][i*M -1] = 0.
                if i < N-1:
                    Qdiags[2][i*M-1] = 0.
            return Qdiags
        PA1tridiag = sp.diags(Pdiags(A1),(-1,0,1),format="csr")
        PA2tridiag = sp.diags(Pdiags(A2),(-1,0,1),format="csr")
        QA1tridiag = sp.diags(Qdiags(A1),(-1,0,1),format="csr")
        QA2tridiag = sp.diags(Qdiags(A2),(-1,0,1),format="csr")

        def firsthalfstep(colTn):
            return sp.linalg.spsolve(PA1tridiag,QA2tridiag.dot(colTn))
        def secondhalfstep(rowTnpoh):
            return sp.linalg.spsolve(QA1tridiag,PA2tridiag.dot(rowTnpoh))

        def step(colTn):
            v = firsthalfstep(colTn)
            return secondhalfstep(v)
        
        def colwrap(T):
            u = np.zeros(L)
            for j in range(M):
                for i in range(N):
                    u[i*M+j] = T[j][i]
            return u
        def colunwrap(wrapped):
            T = np.zeros((M,N))
            for i in range(N):
                for j in range(M):
                    T[j][i] = wrapped[i*M+j]
            return T

        current = colwrap(diffusion_matrix)
        for _ in range(steps):
            current = step(current)
        
        return colunwrap(current)


    def spawn():
        (i,j) = index(crcl[np.random.choice(len(crcl))])
        tries = 0
        while (coord((i,j)) in aggregate
        or diffusion_matrix[j][i] == 1):
            if tries > 10:
                break #there are likely too many particles on the map at once. lower N.
            (i,j) = index(crcl[np.random.choice(len(crcl))])
            tries += 1
        diffusion_matrix[j][i] = 1.

    def init():
        for _ in range(N):
            spawn()
        aggregate.add((0,0))

    

    directions = [(0,1),(1,0),(-1,0),(0,-1)]
    def neighbors(idx):
        i,j = idx
        ret = []
        for dir in directions:
            ia,ja = i+dir[0],j+dir[1]
            if (ia >= 0 and ia <= lattice_sidel 
            and ja >= 0 and ja <= lattice_sidel):
                ret += [(ia,ja)]
        return ret


    init()
    kt = 1 #num steps to evolve the matrix via diffusion before collapsing.
    ptcl_count = 0
    while ptcl_count < desired_ptcls:
        diffusion_matrix = ADI(kt)
        collapse()
        lost = 0
        #remove illegal positions
        for crd in aggregate:
            i,j = index(crd)
            if diffusion_matrix[j][i] == 1:
                diffusion_matrix[j][i] = 0
                lost += 1

        #we want fractal tendrils to emerge. if the particles
        #that would stick are adjacent to one another, the order 
        #they should stick likely can't be determined. 
        #so we just respawn them
        markfordeath = set()
        for crd in aggregate:
            for i,j in neighbors(index(crd)):
                if diffusion_matrix[j][i] - 1 < 0.01:
                    found = []
                    for i2,j2 in neighbors((j,i)):
                        if diffusion_matrix[j2][i2]:
                            found += [(i2,j2)]
                    if len(found):
                        markfordeath.add((i,j))
                        for v in found:
                            markfordeath.add(v)
        for idx in markfordeath:
            i,j = idx
            diffusion_matrix[j][i] = 0

        #the remaining particles are able to stick
        toadd = []
        for crd in aggregate:
            for i,j in neighbors(index(crd)):
                if diffusion_matrix[j][i] - 1 < 0.01 and unif() < sticking_prb:
                    toadd += [coord((i,j))]
                    diffusion_matrix[j][i] = 0.
                    lost += 1
                    ptcl_count += 1
        for v in toadd:
            aggregate.add(v)
    
        #respawn the lost particles
        for _ in range(lost):
            spawn()

                
    if SHOWIMG:
        asimg(lattice_sidel,list(aggregate))
    return aggregate


def anisotropy(method, times):
    def polar_densities(ptcls):
        radial_sections = radius_spawn_ring
        angular_sections = 1000 
        radperbin = 2*np.pi / angular_sections

        def angle(coord):
            agl = np.arctan2(coord[1],coord[0])
            if agl < 0:
                agl += 2*np.pi
            return agl
        def angle2(crcl, i):
            #supports wrapping around more than 2pi rad
            return angle(crcl[i%len(crcl)])+i//len(crcl)* 2 * np.pi
        def dtheta(crcl, i):
            #like typing on a cell keypad with fat fingers, dth
            #is included to indicate how many bins th spans
            agl1 = angle(crcl[(i-1) % len(crcl)]) + (i-1)//len(crcl) * 2 * np.pi
            agl2 = angle(crcl[(i+1) % len(crcl)]) + (i+1)//len(crcl) * 2 * np.pi
            return (agl2-agl1)/2

        def CoM():
            xt = 0
            yt = 0
            for x,y in ptcls:
                xt += x
                yt += y
            return (xt/len(ptcls),yt/len(ptcls))
        com = CoM()

        def adjust_ptcls():
            ret = []
            for x,y in ptcls:
                ret += [(int(x-com[0]),int(y-com[1]))]
            return ret

        agg = adjust_ptcls()
        plr_dns = np.zeros(angular_sections)
        for r in range(1,radial_sections): #the distance, in pixels, is also the index
            c = circle(r)
            c = sorted(c,key=lambda v: angle(v))
            for sec in range(angular_sections):
                theta = sec*radperbin
                ntheta = (sec+1)*radperbin
                cifromrsec = lambda s: int(s*radperbin // (2*np.pi/len(c)))
                px = 0
                i = cifromrsec(sec)
                j = i
                while j < cifromrsec(sec+1):
                    if c[j] in agg: 
                        px += 1
                    j += 1
                found = j-i
                dA = max(found-1,0)
                da = px
                #account for the partial pixel areas on either end of the arc
                lowextra = angle2(c,i)+dtheta(c,i)/2-theta # the angle swept by the top of pixel c[i] down to theta
                highextra = ntheta - (angle2(c,j)-dtheta(c,j)/2) # the angle swept by the bottom of pixel c[j] up to ntheta
                
                dA += lowextra+highextra
                da += lowextra if c[i] in agg else 0
                da += highextra if c[j%len(c)] in agg else 0
                if dA >0.01: #occasional floating point singularties may exist for a large number of angular sections,
                            #when some thin polar section overlaps the boundary of two pixels. We'll just say this area is
                            #negligible, since its hugely unlikely if not impossible that all of the
                            # remaining polar densities in this angular section will do so as well. 
                    plr_dns[sec] += da/dA
            
        return plr_dns

    global SHOWIMG
    SHOWIMG=False
    pdsum = None
    for _ in range(times):
        aggregate = method()
        p = polar_densities(aggregate)
        if pdsum is None:
            pdsum = p
        else:
            pdsum += p
    pd = polar_densities(aggregate)
    s = sum(pd)
    if s < 0.1:
        raise ValueError("aggregates turned out bad")
    normed = np.zeros(len(pd))
    for i,v in enumerate(pd):
        normed[i] = v/s
    return np.var(normed)



def final():
    if SHOWIMG:
        gaussian_rdm_walk()
        diffusion_collapse()
    trials = 3
    repspertrial = 100
    results = []
    for t in range(trials):
        print("trial " + str(t))
        resultsA = anisotropy(gaussian_rdm_walk,repspertrial)
        resultsB = anisotropy(diffusion_collapse,repspertrial)
        results += [(resultsA,resultsB)]
    print(repr(results))


#final()
diffusion_collapse()
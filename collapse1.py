from util import *

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
    kt = 5 #num steps to evolve the matrix via diffusion before collapsing.
    ptcl_count = 0
    lpc = ptcl_count
    while ptcl_count < desired_ptcls:
        if (ptcl_count >= 10+lpc):
            print(str(ptcl_count)+" ptcls")
            ptcl_count = lpc
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
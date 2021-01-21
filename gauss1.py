from util import *

def gaussian_rdm_walk():
    print("gaussian_rdm_walk")    
    aggregate = set([(0,0)])
    crcl = circle(radius_spawn_ring)
    #asimg(lattice_sidel,crcl) #is in fact a circle

    def spawn():
        ch = np.random.choice(range(len(crcl)))
        return crcl[ch]
    
    def gauss2():
        return (rnorm(),rnorm())

    directions = [(0,1),(0,-1),(1,0),(-1,0)]
    ptcl = None
    num_ptcls = 0
    lpc = num_ptcls
    while len(aggregate) < desired_ptcls:
        if num_ptcls >= 10+lpc:
            print(num_ptcls)
            lpc = num_ptcls
        
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
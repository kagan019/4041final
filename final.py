from util import *
from gauss1 import *
from gauss2 import *
from collapse1 import *
from collapse2 import *


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



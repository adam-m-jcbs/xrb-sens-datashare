#! /usr/bin/python
"""
Python replacement for exp.cpp explosion finder.
"""





import sys
import os
import math
import glob
import numpy as np

from kepdump import loaddump
from physconst import XMSUN_KEPLER

alpha_accuracy = 2
pattern = "xxx"
generator_tempelete = ""
base_name = ""
kepler = ""

def alpha2string(val):
    return ("{:0."+str(alpha_accuracy)+"f}")\
        .format(val / 10**alpha_accuracy)

def generator(alpha, link):
    alpha_string = alpha2string(alpha)
    link_file_name = link + ".link"
    with open(generator_tempelete,"r") as fin,\
            open(link_file_name,"w") as fout:
        for line in fin:
            fout.write(line.replace(pattern,alpha_string))

def make_cmd(link):
    # also copy exp.cmd file at the end of cmd file if it exists
    exp_cmd_file = "exp.cmd"
    cmd_file = base_name + link + ".cmd" 
    with open(cmd_file,"w") as fcmd:
        fcmd.write("link {}.link\n".format(link))
        if os.path.exists(exp_cmd_file):
            with open("exp.cmd","r") as fexpcmd:
                fcmd.write(\
"""
; commands from exp.link
""")
                for line in fexpcmd:
                    fcmd.write(line)

def getfallback(link, dump):
    dump_file = base_name + link
    if (dump == 0):
        dump_file += "#envel"
    else:
        dump_file += "#final"
    fail = not os.path.exists(dump_file)
    if not fail:
        dump = loaddump(dump_file)
        xbind = dump.xbind()
        rn = dump.rn
        un = dump.un
        uesc = dump.uesc()
        zone = np.argwhere((xbind > 0.) * (rn > 1.e10) * (un > 0.01 * uesc))[0][0]
        mass = dump.zm()[zone]
    return not fail, zone, mass

def linkplus(link):
    val = 0
    for b in bytearray(link):
        val *= 51
        if 96 < b < 103: # g is not allowd
            val += b - 97
        elif 103 < b < 123:
            val += b - 98
        elif 64 < b < 91:
            val += b - 39
        else:
            raise KeyError("[LINKPLUS] error: " + link)
    print("val = {}".format(val))
    val += 1
    link = ''
    while val > 0:
        val, c = divmod(val, 51)
        if c > 26:
            link = chr(c + 39) + link
        elif c < 6:
            link = chr(c + 97) + link
        else:
            link = chr(c + 98) + link
    return link

class State(object):
    def __init__(self, alpha):
        self.val_low  = alpha
        self.val_hi   = alpha
        self.xval_low = -1.e99
        self.xval_hi  = -1.e99

    def initialize(self, xval):
        if self.xval_low < 0:
            self.xval_low = xval
        if self.xval_hi < 0:
            self.xval_hi = xval
        

def nextalpha(state, val, xval, xval_goal, xval_use, direction):
    fac = 1.5
    fac_max = 10./3.;
    fac_ext = 4./3.;
    val_good = 0.75;

  # print(" [DEBUG] val = {}, val_low = {}, val_hi = {}".format(val, val_low, val_hi))
  # print(" [DEBUG] xval_goal = {}, xval_low = {}, xval_hi = {}, xval_use = {}".format(xval_goal, xval_low, xval_hi, xval_use))
  # print(" [DEBUG] xval = {}, direction = {}".format(xval, direction))

    if (direction == 0):
        print(" [NEXTALPHA] done: direction = {}".format(direction))
        return True, Ellipsis

  # initialize
    state.initialize(xval)

  # cases...
    if direction < 0: 
        if state.val_low == val:
            valx = val / fac
            state.xval_low = xval
            if val_low != val_hi:
                dx = state.xval_hi - state.xval_low
                if dx > 0:
                    print(" [NEXTALPHA] using extrapolation")
                    valx = (xval_goal - state.xval_low) * (state.val_hi - state.val_low) / dx + state.val_low
                    if valx > val / fac:
                        valx = val / fac
                        print(" [NEXTALPHA] limiting extrapolation")
                    if valx < val / fac_max:
                        valx = val / fac_max
                        print(" [NEXTALPHA] limiting extrapolation")
            state.val_hi = val
            state.xval_hi = xval
            val = int(math.ceil(valx))
            if val >= state.val_low: 
                val -= 1
            if val <= 0:
                print(" [NEXTALPHA] done: state.val_low = {}".format(state.val_low))
                return True, val
            state.val_low = val
            state.xval_low = xval
            return False, val
        else:
            state.val_hi = val
            state.xval_hi = xval
            if state.val_hi == state.val_low + 1:
                print(" [NEXTALPHA] done: state.val_low + 1 = state.val_hi = {}".format(state.val_hi))
                return True, val
            dx = state.xval_hi - state.xval_low
            if (dx/state.xval_hi < xval_use) and (dx > 0.):
                print(" [NEXTALPHA] using interpolation")
                valx = (xval_goal - state.xval_low) * (state.val_hi - state.val_low) / dx + state.val_low
                # if valx < 0.5 * (state.val_low + state.val_hi):
                if valx - math.floor(valx) > 0.5:
                    val = int(math.ceil(valx))
                else:
                    val = int(math.floor(valx))
                if val <= state.val_low: 
                    val = state.val_low + 1
                if (val >= state.val_hi):
                    val = state.val_hi - 1
                # terminate if last estimate (state.val_hi) was good enough
                if val == state.val_hi - 1:
                    xval1 = ((val - state.val_low) // (state.val_hi - state.val_low)) * dx + state.xval_low
                    if  (xval - xval_goal  >= 0) and\
                            (xval_goal - xval1 >= 0) and\
                            ((xval - xval_goal) < val_good * (xval_goal - xval1)):
                        val = state.val_hi
                        print(" [NEXTALPHA] done: good val = {}".format(val))
                        return True, val
            else:
                val = int((state.val_low + state.val_hi) / 2)
    else:
        if state.val_hi == val:
            valx = val * fac
            state.xval_hi = xval
            if state.val_low != state.val_hi:
                dx = state.xval_hi - state.xval_low
                if dx > 0:
                    print(" [NEXTALPHA] using extrapolation")
                    valx = (xval_goal - state.xval_low) * (state.val_hi - state.val_low) * fac_ext / dx + state.val_low
                    if valx < val * fac:
                        valx = val * fac
                        print(" [NEXTALPHA] limiting extrapolation")
                    if valx > val * fac_max:
                        valx  =val * fac_max
                        print(" [NEXTALPHA] limiting extrapolation")
            state.val_low = val
            state.xval_low = xval
            val = int(math.floor(valx))
            if val == state.val_hi:
                val += 1
            state.val_hi = val
            state.xval_hi = xval
            return False, val
        else:
            state.val_low = val
            state.xval_low = xval
            if state.val_hi == state.val_low + 1:
                print(" [NEXTALPHA] done: state.val_low + 1 = state.val_hi = {}".format(state.val_hi))
                return True, val
            dx = state.xval_hi - state.xval_low;
            if (dx / state.xval_hi < xval_use) and (dx > 0):
                print(" [NEXTALPHA] using interpolation")
                valx = (xval_goal - state.xval_low) * (state.val_hi - state.val_low) / dx + state.val_low
                # if valx < 0.5 * (state.val_low + state.val_hi)
                if valx - math.floor(valx) > 0.5:
                    val = int(math.ceil(valx))
                else:
                    val = int(math.floor(valx))
                if (val <= state.val_low):
                    val = state.val_low + 1
                if (val >= state.val_hi):
                    val = state.val_hi - 1  
                # terminate if last estimate (state.val_low) was good enough
                if val == state.val_low+1:
                    xval1 = ((val - state.val_low) // (state.val_hi - state.val_low)) * dx + state.xval_low
                    if  (xval_goal-xval  >= 0) and\
                            (xval1 - xval_goal >= 0) and\
                            ((xval_goal - xval) < val_good * (xval1 - xval_goal)):
                        val = state.val_low;
                        print(" [NEXTALPHA] done: good val = {}".format(val))
                        return True, val
            else:
                val = int((state.val_low + state.val_hi) / 2)
    if val <= state.val_low:
            val = state.val_low + 1;
    if val >= state.val_hi:
            val = state.val_hi - 1;
    return False, val


def getresult(link, zone):
    ekin=0.
    mni=0.
    dump_file = base_name + link + "#final"
    fail = not os.path.exists(dump_file)
    if not fail:
        dump = loaddump(dump_file)
        ekin = dump.qparm.enk
        mni = np.sum(dump.xm[zone:-1]*dump.ni56()[zone:-1])/XMSUN_KEPLER
    return not fail, mni, ekin 

def result(alpha, link, mni, ekin, mass_cut, zone_cut):
    temp_link_name = "xxx" + link + ".link"
    link_file_name = link + ".link"
    alpha_string = alpha2string(alpha)
    with open(link_file_name,'r') as fin,\
            open(temp_link_name,'w') as fout:
        fout.write("""c # version 1.00.00
c # automatically generated comment
c # ---------------------------------
c #  RESULT:
c # ---------------------------------
c #  alpha:    {}
c #  mni:      {}
c #  ekin:     {}
c #  mass cut: {}
c #  zone cut: {}
c # ---------------------------------
c # original file follows below this line
""".format(alpha_string,
           mni,
           ekin,
           mass_cut,
           zone_cut))
        for line in fin:
            fout.write(line)
    os.remove(link_file_name)
    os.rename(temp_link_name,link_file_name)

def explosion():
    run = True;
    force = False;
    
    for arg in sys.argv[1:]:
        if arg == "log":
            run = False
        elif arg == "force":
            force = True

    global alpha_accuracy, generator_tempelete, base_name, kepler

            
    #read name of program
    kepler = input()
 
    # read name of explosion generator templete
    presn_dump_name = input()
  
    # read base name for explosion
    base_name = input()
  
    # read name of explosion generator templete
    generator_tempelete = input()
  
    # read start name explosion generator
    generator_start = input()

    # read start piston value
    alpha_start = float(input())

    # read start piston value
    alpha_accuracy = int(input())

    # read goal flag
    flag = input()

    # read goal value
    val = float(input())

    # init alpha iterations
    alpha = int(alpha_start * 10**alpha_accuracy)
    state = State(alpha)

    # init generator name
    link = generator_start

    alpha_best = -1
    link_best = "#"
    best = 1.e99;
    zone_best = 0
    mass_best = 0

    # echo start calulation
    print("{}: first alpha = {}".format(link, alpha))


    # set up limits for mod. reg. falsi algorithm for solution finder
    if flag == "mni":
        val_use = 0.2
    else:
        val_use = 10.


    # start of main loop
    done = False
    while not done:
        if run:
            finished = True

            # look if #final dump is present      
            dump_file_name = base_name + link +"#final"
            if not os.path.exists(dump_file_name):
                finished = False

            # look if #envel dump is present
            dump_file_name = base_name + link +"#envel"
            if not os.path.exists(dump_file_name):
                finished = False

            # all necessary files exist
            if (finished == False) or (force == True):
                # set up first part of call
                call_string = "{} {}{} ".format(kepler, base_name, link)
    
                # look if a restart (z) dump is present
                dump_file_name = base_name + link + "z" 
                if (not os.path.exists(dump_file_name)) or (force == True):
                    print("RUN")
                  
                    # make generator
                    generator(alpha, link)
                    
                    # make command file
                    make_cmd(link);
                    
                    # start from #presn dump
                    call_string += presn_dump_name
                else:
                    print("CONTINUE")
                                  
                    # set "restart" option for call
                    call_string += "z"
                # redirect IO
                call_string += " k </dev/null >/dev/null"
                
                # call KEPLER
                print("CALL: {}".format(call_string))
                os.system(call_string)
            else:
                print("FINISHED")

        # analyze the result
        ok, zone, mass = getfallback(link, 0)
        if not ok:
            print(" [EXPLOSION] ERROR in fallback determination.")
            return 1
        print("zone = {}, mass = {}".format(zone,mass))
        zone_envel = zone
        mass_envel = mass
        ok, zone, mass = getfallback(link, 1);
        if not ok:
            print(" [EXPLOSION] ERROR in fallback determination.")
            return 1        
        if (zone_envel < zone):
            print(" [EXPLOSION] LATE FALLBACK")
            print("zone = {}, mass = {}".format(zone,mass))
        else:
            zone = zone_envel
            mass = mass_envel
        ok, mni, ekin = getresult(link, zone)
        if not ok:
            print(" [EXPLOSION] ERROR in result determination.")
            return 1
        print("mni = {}, ekin = {}".format(mni, ekin))
        result(alpha, link, mni, ekin, mass, zone)

        # determine next piston value
        if flag == "mni":
            val_current = mni
        else:
            val_current = ekin
        direction = 0
        if (val > val_current): 
            direction = +1
        if (val < val_current): 
            direction = -1

        dist = direction * (val - val_current)
        if (dist < best): 
            best = dist
            val_best = val_current
            alpha_best = alpha
            link_best = link
            zone_best = zone
            mass_best = mass

        print("{},{},{}".format(val,
                                val_current,
                                direction))

        done, alpha = nextalpha(state, 
                                alpha, 
                                val_current,    
                                val, 
                                val_use,
                                direction)

        if not done:
            link = linkplus(link);
            print()
            print("{}: new alpha = {} ({},{})".format(link, 
                                                      alpha, 
                                                      state.val_low, 
                                                      state.val_hi))

    call_string = alpha2string(alpha_best)
    print("""

---------------------------------
 RESULT:
---------------------------------
 goal:       {} = {}
 best value: {} = {}
 deviation:  {}
 best alpha: {}
 in link:    {}.link
 mass cut:   {}
 zone cut:   {}
---------------------------------


done.""".format(flag, val,
                flag, val_best, 
                best,
                call_string, 
                link_best,
                mass_best,
                zone_best))

    # clean up
    for filename in glob.iglob("xxx*"):
        os.remove(filename)

if __name__ == "__main__":
    explosion()  

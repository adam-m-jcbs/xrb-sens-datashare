import numpy as np
import stardb
import kepdump
import isotope
import os

def data(dbfilename = os.path.expanduser('~/python/project/znuc2012.S4.star.el.y.stardb.gz')):
    """
    This is the main data collecting module which gets every single isotope/remnant mass from the database which is later used to interpolate from to obtain desired values
    """
    db = stardb.load(dbfilename)                 # loads database
    nmass = db.nvalues[0]                         # finds the number of values
    masses = db.values[0][:nmass]                 #creates a vector of the initial masses
    isodb = stardb.load(os.path.expanduser('~/python/project/znuc2012.S4.star.deciso.y.stardb.gz'))
    
    massnumber = []
    for x in range(len(isodb.ions)):
        mn = isodb.ions[x].A
        massnumber.append(mn)
    massnumber = np.array(massnumber)
    np.save(os.path.expanduser('~/python/project/filestoload/Massnumber'), massnumber)  
#######################            
# write all energy and mixing values

    energyvalues = np.unique(db.fielddata['energy'])
    mixingvalues = np.unique(db.fielddata['mixing'])
    masterremnant = [] # result will be a multidimensional array
    elementdata = []
    isodata = []
    r = len(db.ions)  # for loop iteration
    w = len(isodb.ions)
    for energy in energyvalues:
        remmixingarray = [] # reinitialise the next dimension
        elmixingarray = []
        isomixingarray = []
        for mixing in mixingvalues:
    
        
            ii = np.logical_and(np.isclose(db.fielddata['energy'], energy), np.isclose(db.fielddata['mixing'], mixing))
            
            mass = db.fielddata[ii]['remnant']
            remmixingarray.append(mass) # this is an array of remnant masses for one energy and every mixing value
            
            elfill = [] # reinitialise the next dimension again
            isofill = []
            
            
            for m in range(w):
        
                a = isodb.ions[m]  #for obtaining the element string
                kk = np.where(isodb.ions==isotope.ion(a))  # finding the indices in db.ions for a particular element
                jj = np.where(ii)
                isotopes = isodb.data[jj, kk][0]  # array of abundances for that particular element
                isofill.append(isotopes) # this is an array of element data for every mass for one energy and one mixing value




            isomixingarray.append(isofill)    
            
       
        masterremnant.append(remmixingarray) # these master arrays have every bit of data under its own energy. so called like elementdata[energy][mixing][elementnumber] gives the element data for every star for a single element.
        
        isodata.append(isomixingarray)
    
    np.save(os.path.expanduser('~/python/project/filestoload/IsoData'), isodata)
    np.save(os.path.expanduser('~/python/project/filestoload/RemnantMasses'), masterremnant)
    np.save(os.path.expanduser('~/python/project/filestoload/Ioninfo'), isodb.ions)
    time = []                                     
       
    for mass in masses:   # for loop will cycle through the masses and grab the lifetime of each star
        s = str(mass)     # converts the mass number to a string for file acquiring
        if s.endswith('.0'):    # formatting issue, to match the filenames
            s = s[:-2]                            
        filename =  os.path.expanduser('~/python/project/dumps/z{}#presn').format(s)
        # grabs filename corrosponding to this mass
        d = kepdump.load(filename)                # loads the kepdump data for this star
        time.append(d.time)                     
    yr = 365.2425*86400        
    time = np.array(time)/yr
    dataarray = [masses, time]


    return dataarray

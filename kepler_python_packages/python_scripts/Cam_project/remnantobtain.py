import numpy as np        



def remnant(energyvalues, mixingvalues, energy, mixing, rem, count):
   
    # count 2 is for iterating through the different elements, count is for each star
    
    """
    This program linearly interpolates the remnant masses to grab the correct one for a given energy and mixing value
    """
    if energy > energyvalues[-1]:
        energy = energyvalues[-1]
    elif energy < energyvalues[0]:
        energy = energyvalues[0]
    elif mixing > mixingvalues[-1]:
        mixing = mixingvalues[-1]
    elif mixing < mixingvalues[0]:
        mixing = mixingvalues[0]
        
    # checks if the enegry or mixing values is exact    
    enlogical = np.isclose(energyvalues, energy)
    mixlogical = np.isclose(mixingvalues, mixing)
    
    if enlogical.any() and mixlogical.any(): # both exact, no interpolation 
        enindex = np.where(enlogical)[0][0]
        mixindex = np.where(mixlogical)[0][0]
        
        remnantmass = rem[enindex][mixindex][count]
            
    elif mixlogical.any(): # just mixing is exact, use energy difference for scaling
        mixindex = np.where(mixlogical)[0][0]
        enupper = np.where(energyvalues>energy)[0][0] 
        enlower = np.where(energyvalues<energy)[0][-1]
        
        scale = (energy - energyvalues[enlower])/(energyvalues[enupper] - energyvalues[enlower])
        remnantupper = rem[enupper][mixindex][count] # grabs upper remnantmass
        remnantlower = rem[enlower][mixindex][count] # grabs lower remnantmass
        remnantmass = remnantlower + scale*(remnantupper - remnantlower) # interpolates b/w the two
            
    elif enlogical.any(): # just energy exact, se mixing difference for scaling
        enindex = np.where(enlogical)[0][0]
        mixupper = np.where(mixingvalues>mixing)[0][0]
        mixlower = np.where(mixingvalues<mixing)[0][-1]
        
        scale = (mixing - mixingvalues[mixlower])/(mixingvalues[mixupper] - mixingvalues[mixlower])
        remnantupper = rem[enindex][mixupper][count] # grabs upper remnantmass
        remnantlower = rem[enindex][mixlower][count] # grabs lower remnantmass
        remnantmass = remnantlower + scale*(remnantupper - remnantlower) # interpolates b/w the two
        
            
    else:   # neither exact
        mixupper = np.where(mixingvalues>mixing)[0][0]
        mixlower = np.where(mixingvalues<mixing)[0][-1]
        enupper = np.where(energyvalues>energy)[0][0] 
        enlower = np.where(energyvalues<energy)[0][-1]
        

        
        enscale = (energy - energyvalues[enlower])/(energyvalues[enupper] - energyvalues[enlower])
        mixscale = (mixing - mixingvalues[mixlower])/(mixingvalues[mixupper] - mixingvalues[mixlower])
        
        remnant1 = rem[enupper][mixupper][count] # grabs enupper mixupper remnantmass
        remnant2 = rem[enlower][mixlower][count] # grabs enlower mixlower remnantmass
        remnant3 = rem[enupper][mixlower][count] # grabs enupper mixlower remnantmass
        remnant4 = rem[enlower][mixupper][count] # grabs enlower mixupper remnantmass
        
        # interpolate between each pair individually like done above to obtain 4 interpolated remnant masses. Find the average of these four.
        
        av1 = remnant2 + enscale*(remnant3 - remnant2)
        av2 = remnant3 + mixscale*(remnant1 - remnant3) 
        av3 = remnant4 + enscale*(remnant1 - remnant4) 
        av4 = remnant2 + mixscale*(remnant4 - remnant2)
        
        remnantmass = 0.25*(av1 + av2 + av3 + av4) 
    return remnantmass
    
















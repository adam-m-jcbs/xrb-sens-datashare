import numpy as np        



def element(energyvalues, mixingvalues, energy, mixing, data, count, count2):
    # count 2 is for iterating through the different elements, count is for each star
    # this section is for checking is the energy or mixing values are outside our range

    """
    This module interpolates to grab the correct isotope yields for the given energy and mixing values
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
        
        element = data[enindex][mixindex][count2][count]
            
    elif mixlogical.any(): # just mixing is exact, use energy difference for scaling
        mixindex = np.where(mixlogical)[0][0]
        enupper = np.where(energyvalues>energy)[0][0] 
        enlower = np.where(energyvalues<energy)[0][-1]
        
        scale = (energy - energyvalues[enlower])/(energyvalues[enupper] - energyvalues[enlower])
        elementupper = data[enupper][mixindex][count2][count] # grabs upper element
        elementlower = data[enlower][mixindex][count2][count] # grabs lower element
        element = elementlower + scale*(elementupper - elementlower) # interpolates b/w the two
            
    elif enlogical.any(): # just energy exact,  mixing difference for scaling
        enindex = np.where(enlogical)[0][0]
        mixupper = np.where(mixingvalues>mixing)[0][0]
        mixlower = np.where(mixingvalues<mixing)[0][-1]
        
        scale = (mixing - mixingvalues[mixlower])/(mixingvalues[mixupper] - mixingvalues[mixlower])
        elementupper = data[enindex][mixupper][count2][count] # grabs upper element
        elementlower = data[enindex][mixlower][count2][count] # grabs lower element
        element = elementlower + scale*(elementupper - elementlower) # interpolates b/w the two
        
            
    else:   #neither exact 
        mixupper = np.where(mixingvalues>mixing)[0][0]
        mixlower = np.where(mixingvalues<mixing)[0][-1]
        enupper = np.where(energyvalues>energy)[0][0] 
        enlower = np.where(energyvalues<energy)[0][-1]

        enscale = (energy - energyvalues[enlower])/(energyvalues[enupper] - energyvalues[enlower])
        mixscale = (mixing - mixingvalues[mixlower])/(mixingvalues[mixupper] - mixingvalues[mixlower])
        
        element1 = data[enupper][mixupper][count2][count] # grabs enupper mixupper element
        element2 = data[enlower][mixlower][count2][count] # grabs enlxower mixlower element
        element3 = data[enupper][mixlower][count2][count] # grabs enupper mixlower element
        element4 = data[enlower][mixupper][count2][count] # grabs enlower mixupper element
        
        # interpolate between each pair individually like done above to obtain 4 interpolated element. Find the average of these four.
        
        av1 = element2 + enscale*(element3 - element2)
        av2 = element3 + mixscale*(element1 - element3) 
        av3 = element4 + enscale*(element1 - element4) 
        av4 = element2 + mixscale*(element4 - element2)
        
        element = 0.25*(av1 + av2 + av3 + av4) 
    return element

"""
hf 181 fit attempt
"""

from physconst import EV, KB

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

E = np.array([0, 68.e3]) * EV
j = np.array([1/2, 9/2])
r = np.array([1.900E-07, 2.2368e-05])


E = np.array([0, 62.e3]) * EV
j = np.array([1/2, 9/2])
r = np.array([1.900E-07, 2.16312e-05])

# E = np.array([0.,46, 99]) * 1e3*EV
# j = np.array([1/2,3/2,5/2])
# r = np.array([1.900E-07,.8e-5, 6.e-5])


g = 2 * j + 1
def rate(t):
    if not isinstance(t, np.ndarray):
        t = np.array([t])
    n = g[:, np.newaxis] * np.exp(-E[:,np.newaxis] / (KB * t[np.newaxis,:]))
    return np.sum(r[:,np.newaxis] * n, axis = 0) / np.sum(n, axis = 0)


def plot():
    t= np.logspace(7,10,100)
    r = rate(t)

    f = plt.figure()
    ax = f.add_subplot(111)

    ax.plot(t,r)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('T/K')
    ax.set_ylabel('rate/Hz')


    ax.plot(data[0],data[1],'+')

    plt.draw()


data = [[0.0100, .1899E-06],
        [0.0103, .1900E-06],
        [0.0106, .1900E-06],
        [0.0109, .1900E-06],
        [0.0112, .1900E-06],
        [0.0115, .1900E-06],
        [0.0118, .1901E-06],
        [0.0121, .1901E-06],
        [0.0125, .1901E-06],
        [0.0128, .1901E-06],
        [0.0132, .1902E-06],
        [0.0136, .1902E-06],
        [0.0140, .1902E-06],
        [0.0143, .1903E-06],
        [0.0147, .1903E-06],
        [0.0152, .1903E-06],
        [0.0156, .1903E-06],
        [0.0160, .1904E-06],
        [0.0165, .1904E-06],
        [0.0169, .1904E-06],
        [0.0174, .1905E-06],
        [0.0179, .1905E-06],
        [0.0184, .1905E-06],
        [0.0189, .1906E-06],
        [0.0195, .1906E-06],
        [0.0200, .1907E-06],
        [0.0206, .1907E-06],
        [0.0211, .1907E-06],
        [0.0217, .1908E-06],
        [0.0224, .1908E-06],
        [0.0230, .1909E-06],
        [0.0236, .1909E-06],
        [0.0243, .1910E-06],
        [0.0250, .1910E-06],
        [0.0257, .1911E-06],
        [0.0264, .1911E-06],
        [0.0271, .1912E-06],
        [0.0279, .1912E-06],
        [0.0287, .1913E-06],
        [0.0295, .1913E-06],
        [0.0303, .1914E-06],
        [0.0312, .1915E-06],
        [0.0321, .1915E-06],
        [0.0330, .1916E-06],
        [0.0339, .1917E-06],
        [0.0348, .1917E-06],
        [0.0358, .1918E-06],
        [0.0368, .1919E-06],
        [0.0379, .1919E-06],
        [0.0389, .1920E-06],
        [0.0400, .1921E-06],
        [0.0412, .1922E-06],
        [0.0423, .1923E-06],
        [0.0435, .1924E-06],
        [0.0447, .1924E-06],
        [0.0460, .1925E-06],
        [0.0473, .1926E-06],
        [0.0486, .1927E-06],
        [0.0500, .1928E-06],
        [0.0514, .1929E-06],
        [0.0528, .1930E-06],
        [0.0543, .1931E-06],
        [0.0558, .1932E-06],
        [0.0574, .1934E-06],
        [0.0590, .1935E-06],
        [0.0607, .1936E-06],
        [0.0624, .1937E-06],
        [0.0642, .1939E-06],
        [0.0660, .1940E-06],
        [0.0678, .1941E-06],
        [0.0697, .1943E-06],
        [0.0717, .1944E-06],
        [0.0737, .1945E-06],
        [0.0758, .1947E-06],
        [0.0779, .1949E-06],
        [0.0801, .1950E-06],
        [0.0823, .1952E-06],
        [0.0847, .1954E-06],
        [0.0870, .1955E-06],
        [0.0895, .1957E-06],
        [0.0920, .1959E-06],
        [0.0946, .1961E-06],
        [0.0973, .1963E-06],
        [0.1000, .1965E-06],
        [0.1028, .1967E-06],
        [0.1057, .1969E-06],
        [0.1087, .1971E-06],
        [0.1117, .1974E-06],
        [0.1149, .1976E-06],
        [0.1181, .1978E-06],
        [0.1214, .1981E-06],
        [0.1248, .1983E-06],
        [0.1284, .1986E-06],
        [0.1320, .1989E-06],
        [0.1357, .1991E-06],
        [0.1395, .1994E-06],
        [0.1434, .1997E-06],
        [0.1475, .2000E-06],
        [0.1516, .2003E-06],
        [0.1559, .2007E-06],
        [0.1603, .2010E-06],
        [0.1648, .2013E-06],
        [0.1694, .2017E-06],
        [0.1742, .2020E-06],
        [0.1791, .2024E-06],
        [0.1841, .2028E-06],
        [0.1893, .2032E-06],
        [0.1946, .2036E-06],
        [0.2001, .2040E-06],
        [0.2057, .2045E-06],
        [0.2115, .2049E-06],
        [0.2174, .2054E-06],
        [0.2236, .2058E-06],
        [0.2299, .2063E-06],
        [0.2363, .2068E-06],
        [0.2430, .2073E-06],
        [0.2498, .2079E-06],
        [0.2568, .2080E-06],
        [0.2641, .2081E-06],
        [0.2715, .2081E-06],
        [0.2791, .2082E-06],
        [0.2870, .2082E-06],
        [0.2950, .2083E-06],
        [0.3033, .2084E-06],
        [0.3119, .2084E-06],
        [0.3206, .2084E-06],
        [0.3297, .2084E-06],
        [0.3389, .2084E-06],
        [0.3485, .2084E-06],
        [0.3583, .2084E-06],
        [0.3684, .2084E-06],
        [0.3787, .2084E-06],
        [0.3894, .2084E-06],
        [0.4003, .2085E-06],
        [0.4116, .2086E-06],
        [0.4232, .2087E-06],
        [0.4351, .2088E-06],
        [0.4473, .2089E-06],
        [0.4599, .2090E-06],
        [0.4728, .2091E-06],
        [0.4861, .2093E-06],
        [0.4998, .2094E-06],
        [0.5139, .2098E-06],
        [0.5283, .2103E-06],
        [0.5432, .2108E-06],
        [0.5585, .2113E-06],
        [0.5742, .2119E-06],
        [0.5903, .2124E-06],
        [0.6069, .2130E-06],
        [0.6240, .2136E-06],
        [0.6415, .2144E-06],
        [0.6596, .2155E-06],
        [0.6781, .2167E-06],
        [0.6972, .2178E-06],
        [0.7168, .2190E-06],
        [0.7370, .2203E-06],
        [0.7577, .2216E-06],
        [0.7791, .2229E-06],
        [0.8010, .2255E-06],
        [0.8235, .2307E-06],
        [0.8467, .2362E-06],
        [0.8705, .2420E-06],
        [0.8950, .2481E-06],
        [0.9201, .2545E-06],
        [0.9460, .2613E-06],
        [0.9726, .2685E-06],
        [1.0000, .2761E-06],
        [1.0281, .2950E-06],
        [1.0571, .3159E-06],
        [1.0868, .3389E-06],
        [1.1174, .3644E-06],
        [1.1488, .3925E-06],
        [1.1811, .4236E-06],
        [1.2143, .4583E-06],
        [1.2485, .4968E-06],
        [1.2836, .5451E-06],
        [1.3197, .6021E-06],
        [1.3568, .6669E-06],
        [1.3950, .7408E-06],
        [1.4343, .8254E-06],
        [1.4746, .9224E-06],
        [1.5161, .1034E-05],
        [1.5587, .1163E-05],
        [1.6026, .1296E-05],
        [1.6477, .1421E-05],
        [1.6940, .1562E-05],
        [1.7417, .1722E-05],
        [1.7907, .1904E-05],
        [1.8410, .2110E-05],
        [1.8928, .2346E-05],
        [1.9461, .2615E-05],
        [2.0008, .2911E-05],
        [2.0571, .3113E-05],
        [2.1150, .3336E-05],
        [2.1744, .3581E-05],
        [2.2356, .3852E-05],
        [2.2985, .4153E-05],
        [2.3632, .4486E-05],
        [2.4296, .4856E-05],
        [2.4980, .5269E-05],
        [2.5683, .5555E-05],
        [2.6405, .5818E-05],
        [2.7148, .6102E-05],
        [2.7911, .6408E-05],
        [2.8697, .6739E-05],
        [2.9504, .7097E-05],
        [3.0334, .7485E-05],
        [3.1187, .7906E-05],
        [3.2064, .8248E-05],
        [3.2966, .8496E-05],
        [3.3894, .8760E-05],
        [3.4847, .9039E-05],
        [3.5828, .9335E-05],
        [3.6835, .9650E-05],
        [3.7872, .9984E-05],
        [3.8937, .1034E-04],
        [4.0032, .1068E-04],
        [4.1158, .1087E-04],
        [4.2316, .1107E-04],
        [4.3507, .1128E-04],
        [4.4730, .1150E-04],
        [4.5989, .1174E-04],
        [4.7282, .1198E-04],
        [4.8612, .1224E-04],
        [4.9980, .1250E-04],
        [5.1386, .1266E-04],
        [5.2831, .1280E-04],
        [5.4317, .1295E-04],
        [5.5845, .1310E-04],
        [5.7416, .1327E-04],
        [5.9032, .1343E-04],
        [6.0692, .1361E-04],
        [6.2399, .1379E-04],
        [6.4155, .1391E-04],
        [6.5959, .1399E-04],
        [6.7815, .1406E-04],
        [6.9723, .1414E-04],
        [7.1684, .1422E-04],
        [7.3700, .1431E-04],
        [7.5774, .1440E-04],
        [7.7905, .1449E-04],
        ]

data = np.array(data).transpose()
data[0] *= 1.e8

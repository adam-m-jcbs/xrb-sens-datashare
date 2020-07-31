import urllib.request
from bs4 import BeautifulSoup
import isotope
import re
import numpy as np

#def get_data(el = 'C'):
el = 'C'
with urllib.request.urlopen(
    "https://en.wikipedia.org/wiki/Isotopes_of_" + isotope.ion(el).element_name()) as page:
    data = page.read()
soup = BeautifulSoup(data, "html.parser")

x = soup('table', class_="wikitable")
t = x[0]

rows = t('tr')
header = rows[0]
cols = header('th')
cs = [' '.join(c.text.split()).split(' [')[0] for c in cols]

h2 = rows[1]('th')
if len(h2) > 0:
    i0 = 2
    print(h2[0].text)
else:
    i0 = 1

imass = cs.index('isotopic mass (u)')


class HTMLTable(object):
    def __init__(self, table):
        rows = table('tr')
        # get sizes
        r0 = rows[0]
        cells = r0(('th', 'td'))
        ncol = 0
        for ci in cells:
            ncol += self.get_cs(ci)
        nrow = 0
        nr = 1
        for ri in rows:
            c0 = ri(('td','th'))[0]
            if nr > 1:
                nr -= 1
                continue
            nr = self.get_rs(c0)
            nrow += nr
        # get data
        data = np.ndarray((nrow,ncol), dtype=np.object)
        data[()] = None
        print(data.shape)
        for ir, r in enumerate(rows):
            cells = r(('th', 'td'))
            ic = 0
            for cell in cells:
                while data[ir, ic] is not None:
                    ic += 1
                ext = self.get_cell_ext(cell)
                # fill "spanned" region with reference to same object
                for i in range(ir,ir+ext[0]):
                    for j in range(ic,ic+ext[1]):
                        data[i,j] = cell
        self.data = data
        self.nrow = nrow
        self.ncol = ncol

    @classmethod
    def get_cell_ext(cls, cell):
        return (
            cls.get_rs(cell),
            cls.get_cs(cell),
            )

    @staticmethod
    def get_rs(cell):
        rs = cell.get('rowspan')
        if rs is None:
            rs = 1
        else:
            rs = int(rs)
        return rs

    @staticmethod
    def get_cs(cell):
        cs = cell.get('colspan')
        if cs is None:
            cs = 1
        else:
            cs = int(cs)
        return cs

    def __getitem__(self, index):
        return self.data[index]


data = HTMLTable(t)


idata = {}

# cont = 0
# for r in rows[i0:]:
#     cells = r('td')
#     ci = cells[0]
#     if cont > 0:
#         cont -= 1
#         continue
#     ion = isotope.ion(''.join(str(x) for x in ci.contents))
#     print(ion)
#     idata = {}
#     data[ion] = idata
#     if ion.is_isotope():
#         mass = float(re.split('[\(# ]+', cells[imass].text)[0])
#         idata['mass'] = mass

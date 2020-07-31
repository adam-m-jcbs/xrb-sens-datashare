#! /usr/bin/python3

import os.path
from collections import OrderedDict
import textwrap

parameter_zero = {'_':-1}

def make_keppar(path = os.path.join('~','kepler','source')):
    """Create the KEPLER parameter file form the KEPLER source code."""
    import os.path
    import re
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    file = 'kepdat.f'
    file = os.path.join(path,file)
    with open(file) as f:
        s = f.read()

    r = re.compile(r'^ *data *[(]nameparm[(]ii[)],ii=[0-9]*,(?:[0-9]*|nparm)[)] /([^/]*)/',re.MULTILINE + re.DOTALL)
    pnames = ''.join(r.findall(s))
    r = re.compile(r"'([^']*)'",re.MULTILINE + re.DOTALL)
    pnames = r.findall(pnames)

    r = re.compile(r'^ *data *[(]iptype[(]ii[)],ii=[0-9]*,(?:[0-9]*|nparm)[)] /([^/]*)/',re.MULTILINE + re.DOTALL)
    pvalues = ','.join(r.findall(s))
    r = re.compile(r'^.{7}([^\n]*)$',re.MULTILINE + re.DOTALL)
    pvalues = ''.join(r.findall(pvalues))
    pvalues = pvalues.replace(' ','')
    pvalues = [int(px) for px in re.split(',',pvalues)]
    print('nparm  = '+str(len(pvalues)))

    p = OrderedDict(parameter_zero)
    p.update(zip(pnames, pvalues))

    r = re.compile(r'^ *data *[(]nameqprm[(]ii[)],ii=[0-9]*,(?:[0-9]*|nqparm)[)] /([^/]*)/',re.MULTILINE + re.DOTALL)
    qnames = ''.join(r.findall(s))
    r = re.compile(r"'([^']*)'",re.MULTILINE + re.DOTALL)
    qnames = r.findall(qnames)

    r = re.compile(r'^ *data *[(]iqtype[(]ii[)],ii=[0-9]*,(?:[0-9]*|nqparm)[)] /([^/]*)/',re.MULTILINE + re.DOTALL)
    qvalues = ','.join(r.findall(s))
    r = re.compile(r'^.{7}([^\n]*)$',re.MULTILINE + re.DOTALL)
    qvalues = ''.join(r.findall(qvalues))
    qvalues = qvalues.replace(' ','')
    qvalues = [int(qx) for qx in re.split(',',qvalues)]
    print('nqparm = '+str(len(qvalues)))

    q = OrderedDict(parameter_zero)
    q.update(zip(qnames, qvalues))

    # path = os.path.join('~','python','source3')
    path = os.path.dirname(__file__)
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    file = 'keppar_data.py'
    file = os.path.join(path,file)
    # textwrap.fill(s,72,subsequent_indent=' '*4,fix_sentence_endings=True, break_on_hyphens=False)
    with open(file, 'w', encoding = 'utf-8') as f:
        f.write('from collections import OrderedDict\n')
        s = 'p = '+str(p)
        s = textwrap.fill(s, 72, subsequent_indent=' '*4,fix_sentence_endings=True, break_on_hyphens=False)
        f.write(s + '\n')
        s = 'q = '+str(q)
        s = textwrap.fill(s, 72, subsequent_indent=' '*4,fix_sentence_endings=True, break_on_hyphens=False)
        f.write(s + '\n')

if __name__ == "__main__":
    make_keppar()

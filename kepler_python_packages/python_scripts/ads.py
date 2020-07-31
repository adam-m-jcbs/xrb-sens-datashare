#! /usr/bin/python3

"""Get reference from ADS bib string"""

import sys
import textwrap
import re
import urllib.request

def get_ref(adsid = '2003ApJ...591..288H'):
    """Get reference from ADS bib string"""
    a_string=r'ARTICLE'
    p_string=r'INPROCEEDINGS'
    b_string=r'BOOK'

    x_string = (a_string,
                p_string,
                b_string)

    # match just article (get rid of leading head)

    # match fields
    p = re.compile(r'^[ a-zA-Z]*? = .*?(?=^[ a-zA-Z]*? = )',re.MULTILINE + re.DOTALL)
    # print p.findall(d)

    url_template = 'http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode={0}&data_type=BIBTEX&db_key=AST&nocookieset=1'
    url = url_template.format(adsid)

    f = urllib.request.urlopen(url)
    x = f.read()
    f.close()

    x = x.decode()

    ref = ''
    # here we need an extra case for @INPROCEEDINGS
    for i, x_s in enumerate(x_string):
        px = re.compile(r'^@'+x_s+'{.*?^}',re.MULTILINE + re.DOTALL)
        refx = px.findall(x)
        if len(refx):
            m_string = x_string[i]
            ref = refx[0]

    isarticle = m_string == a_string
    isproceeding = m_string == p_string
    isbook = m_string == b_string
    isarxiv = False
    isabstract = False

    if ref == '':
        print('String not found.')
        exit()

    fields = p.findall(ref)
    fn = [re.split(' = ',field)[0].strip() for field in fields]
    fc = [re.split(' = ',field)[1] for field in fields]

    fc = [re.split(' = ',field)[1].replace(',\n','').replace('\n','').replace('\t','') for field in fields]
    for i,c in enumerate(fc):
        if c[0] == '"':
            fc[i]=c[1:-1]
    for i,c in enumerate(fc):
        if c[0] == '{':
            fc[i]=c[1:-1]
    fd = dict(zip(fn, fc))

    if isarticle and fd['journal'] == r'ArXiv e-prints':
        isarticle = False
        isarxiv = True

    if isarticle and fd['journal'] == r'APS Meeting Abstracts':
        isarticle = False
        isabstract = True

    if fd['title'][-1] == '.':
        fd['title']= fd['title'][:-1]
    fd['author'] = fd['author'].replace(' and ',', ')
    if not isbook:
        if not 'pages' in fd:
            fd['pages'] = '--'
        else:
            if fd['pages'][-1] == '+':
                fd['pages']= fd['pages'][:-1]
            if fd['pages'][-1] == '-':
                fd['pages']= fd['pages'][:-1]
            fd['pages'] = fd['pages'].replace('-','--')

    if 'pages' in fd and not 'page' in fd:
        fd['page'] = fd['pages'].split('-')[0]

    fd['author'] = fd['author'].replace('~', ' ').replace('{', '').replace('}', '').replace(r"\'", '').replace(r'\c ', '').replace('\v ', '').replace(r'\"', '')

    fd['title'] = fd['title'].replace(r'\~{}', ' ~ ').replace(r'{$\gamma$}', 'gamma').replace(r'$\lt$=', '<=')

    try:
        fd['pages'] = fd['pages'].replace(r'--', '-')
    except:
        pass

    try:
        fd['booktitle'] = fd['booktitle'].replace('~', ' ')
    except:
        pass

    p = re.compile(r'@'+m_string+'{[^,]+?,',re.MULTILINE + re.DOTALL)
    ads = p.findall(x)[0][len(m_string)+2:len(m_string)+21]
    fd['ads'] = ads

    # get citations from maon pahe
    url = r'http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode={0}'.format(adsid)
    print(url)
    f = urllib.request.urlopen(url)
    x = f.read().decode()
    f.close()

    c = re.findall(r'Citations to the Article \((d+)\)', x)
    if len(c) == 0:
        c = ''
    else:
        assert len(c) == 1
        c = c[0]
    fd['citations'] = c


    # maybe replace known article abbreviations
    abbrev = {
        r'\aap': r'A\&A',
        r'\ssr': r'Space Science Reviews',
        r'\aj': r'AJ',
        r'\apj': r'ApJ',
        r'\nat': r'Nature',
        r'\pasp': r'PASP',
        r'\apjs': r'ApJS',
        r'\apjl': r'ApJL',
        r'\araa': r'ARA\&A',
        r'\mnras': r'MNRAS',
        r'\prd': r'Phys.\ Rev.\ D',
        r'\physrep': r'PhR',
        r'Physical Review Letters': r'PRL',
        r'Astronomical Society of the Pacific Conference Series' : r'ASP Conf.\ Series',
        r'American Institute of Physics Conference Series' : r'AIP Conf.\ Series',
        r'\zap':r'Zeitsch.\ f{\"u}r {Astron.}'
        }

    abbrev = {
        r'\aap': r'Astronomy and Astrophysics',
        r'\ssr': r'Space Science Reviews',
        r'\aj': r'Astronomical Journal',
        r'\apj': r'The Astrophysical Journal',
        r'\nat': r'Nature',
        r'\pasa': r'Publications of the Astronomical Society of Australia',
        r'\pasp': r'Publications of the Astronomical Society of the Pacific',
        r'\apjs': r'The Astrophysical Journal Supplement Seri  es',
        r'\apjl': r'The Astrophysical Journal Letters',
        r'\araa': r'Annual Reviews in Astronomy and Astrophysics',
        r'\mnras': r'Monthly Notices of the Royal Astronomical Society',
        r'\prd': r'Phys.\ Rev.\ D',
        r'\physrep': r'Physics Reports',
        r'Physical Review Letters': r'Physical Review Letters',
        r'Astronomical Society of the Pacific Conference Series' : r'ASP Conf.\ Series',
        r'American Institute of Physics Conference Series' : r'AIP Conf.\ Series',
        r'\zap':r'Zeitsch.\ f{\"u}r {Astron.}'
        }

    bib_format = r'{0[author]} ({0[year]}): "{0[title]}", {0[journal]}, {0[volume]}, {0[pages]}.'

    if isarticle:
        fd['journal']=abbrev.get(fd['journal'],fd['journal'])
        #    bib_format = r'\bibitem{{{0[ads]}}} {0[author]}: {0[title]}. {0[journal]}, \textbf{{{0[volume]}}}, {0[pages]} ({0[year]})'
        #    bib_format = r'\bibitem{{{0[ads]}}} {0[author]} ({0[year]}), {0[journal]}, {{{0[volume]}}}, {0[pages]} '
        #        bib_format = r"\bibitem{{{0[ads]}}} {0[author]}, ``{0[title]}'', \textsl{{{0[journal]}}}, \textbf{{{0[volume]}}}, {0[pages]}, ({0[year]})."
        # bib_format = r"\Ref{{}}{{{0[title]}}}{{{0[author]}}}{{{0[journal]}}}{{{0[volume]}}}{{{0[pages]}}}{{{0[year]}}}{{ARTICLE}}{{{0[citations]}}}\ADS{{{0[ads]}}}"

    elif isproceeding:
        fd['series']=abbrev.get(fd['series'],fd['series'])
        #    bib_format = r'\bibitem{{{0[ads]}}} {0[author]}: {0[title]}.  In: {0[editor]} (eds.) {0[booktitle]}.  {0[series]}, \textbf{{{0[volume]}}}, {0[pages]} ({0[year]})'
        #    bib_format = r'\bibitem{{{0[ads]}}} {0[author]} ({0[year]}),  in: {0[editor]} (eds.) {0[booktitle]}.  {0[series]}, {{{0[volume]}}}, {0[pages]}'
        # bib_format = r"\bibitem{{{0[ads]}}} {0[author]}, ''{0[title]}'', in: {0[editor]} (eds.) \textsl{{{0[booktitle]}}}.  {0[series]}, \textbf{{{0[volume]}}}, {0[pages]}, ({0[year]})."
        # bib_format = r"\Ref{{}}{{{0[title]}}}{{{0[author]}}}{{{0[journal]}}}{{{0[volume]}}}{{{0[pages]}}}{{{0[year]}}}{{PROCEEDINGS}}{{{0[citations]}}}\ADS{{{0[ads]}}}
        # bib_format = r"\bibitem{{{0[ads]}}} {0[author]}, ''{0[title]}'', in: {0[editor]} (eds.) \textsl{{{0[booktitle]}}}.  {0[series]}, \textbf{{{0[volume]}}}, {0[pages]}, ({0[year]}).

        bib_format = r'{0[author]} ({0[year]}): "{0[title]}", in: {0[editor]} (eds.) "{0[booktitle]}". {0[series]}, {0[volume]}, {0[pages]}.'

    elif isbook:
        #    bib_format = r'\bibitem{{{0[ads]}}} {0[author]}: {0[title]}.  {0[booktitle]} ({0[year]})'
        #    bib_format = r'\bibitem{{{0[ads]}}} {0[author]}: ({0[year]}), {0[booktitle]} '
        # bib_format = r"\bibitem{{{0[ads]}}} {0[author]}, ''{0[title]}'',  {0[booktitle]}, ({0[year]})."
        # bib_format = r"\Ref{{}}{{{0[title]}}}{{{0[author]}}}{{{0[journal]}}}{{{0[volume]}}}{{{0[pages]}}}{{{0[year]}}}{{BOOK}}{{{0[citations]}}}\ADS{{{0[ads]}}}"

        bib_format = r'{0[author]} ({0[year]}): "{0[title]}", {0[booktitle]}.'

    elif isarxiv:
        #    bib_format = r'\bibitem{{{0[ads]}}} {0[author]}: {0[title]}.  {0[journal]}, {0[archivePrefix]}:{0[eprint]} ({0[year]})'
        #    bib_format = r'\bibitem{{{0[ads]}}} {0[author]} ({0[year]}),  {0[journal]}, {0[archivePrefix]}:{0[eprint]}'
        # bib_format = r"\bibitem{{{0[ads]}}} {0[author]}, ''{0[title]}'',  \textsl{{{0[journal]}}}, {0[archivePrefix]}:{0[eprint]}, ({0[year]})."
        # bib_format = r"\Ref{{}}{{{0[title]}}}{{{0[author]}}}{{{0[journal]}}}{{accepted DATE}}{{\textsl{{in press}}}}}{{{0[year]}}}{{ARTICLE}}{{{0[citations]}}}\ADS{{{0[ads]}}}"
        pass
    elif isabstract:
        # bib_format = r"\bibitem{{{0[ads]}}} {0[author]}, ''{0[title]}'',  \textsl{{{0[journal]}}}, \textbf{{{0[pages]}}}, ({0[year]})."
        # bib_format = r"{0[Author]} {0[year]}, {0[journal]}, {0[volume], {0[page]}."
        # bib_format = r"\Ref{{}}{{{0[title]}}}{{{0[author]}}}{{{0[journal]}}}{{accepted DATE}}{{\textsl{{in press}}}}}{{{0[year]}}}{{ABSTRACT}}{{{0[citations]}}}\ADS{{{0[ads]}}}"
        pass


    s = bib_format.format(fd)

    # may not need to remove \n and \t before; textwrap will do
    #    return textwrap.fill(s,72,subsequent_indent=' '*8,fix_sentence_endings=True,break_on_hyphens=False)

    return s

if __name__ == '__main__':
    ref = sys.argv[1]
    s = get_ref(ref)
    print(s)

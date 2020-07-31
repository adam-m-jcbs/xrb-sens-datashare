#! /bin/env python3

import sys, glob, subprocess, difflib, os, os.path, re, collections
import string, copy, itertools, operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from logged import Logged


class Tester(Logged):

    def __init__(self, name):
        self.name = name
        submissions_dir = '/home/alex/Class/M40001-python-2014/submisions/'
        samples = glob.glob(os.path.join(submissions_dir, name+'*.py'))
        tests = sorted([os.path.splitext(os.path.basename(s))[0][len(name):] for s in samples])
        self.path = submissions_dir
        self.python = os.path.expanduser('~/Python/bin/python3')
        for test in tests:
            self.functions[test](self, name + test)


    def test_replace(self, test):
        print()
        print('='*72)
        print(test)
        print('-'*72)
        sample_filename = 'text.txt'
        result_filename = 'xxx.txt'
        replace = ['John', 'Pat']
        subprocess.check_call([self.python,
                               os.path.join(self.path, test + '.py'),
                               sample_filename,
                               result_filename,
                               ] + replace)
        with open(sample_filename) as f:
            master = re.sub(r'\b{}\b'.format(replace[0]), replace[1], f.read())
        with open(result_filename) as f:
            result = f.read()
        diff = difflib.unified_diff(master.splitlines(), result.splitlines())
        n = 0
        for d in diff:
            print(d)
            n += 1
        if n == 0:
            print('Result OK.')

    def test_ion(self, test):
        parm = ['O16', 'H3']
        print()
        print('='*72)
        print(test)
        print('-'*72)
        try:
            result = subprocess.check_output(
                [self.python,
                 os.path.join(self.path, test + '.py'),
                 ] + parm,
                stderr = subprocess.STDOUT,
                universal_newlines = True)
        except subprocess.CalledProcessError as e:
            print('Program failed')
            print(e)
            return
        # result = 'F19\n'
        x = Nucleus(parm[0])
        for i in parm[1:]:
            x += Nucleus(i)
        if result.strip() == str(x):
            print('Result OK.')
        else:
            print('Expected: {}'.format(str(x)))
            print('Result:')
            print(result)

    def test_vector(self, test):
        print()
        print('='*72)
        print(test)
        print('-'*72)
        sample_filename = 'vector.txt'
        try:
            result = subprocess.check_output(
                [self.python,
                 os.path.join(self.path, test + '.py'),
                 sample_filename],
                stderr = subprocess.STDOUT,
                universal_newlines = True)
        except subprocess.CalledProcessError as e:
            print('Program failed')
            print(e)
            return
        data = []
        with open(sample_filename, 'rt') as f:
            for line in f:
                try:
                    data += [float(line)]
                except:
                    pass
        array = np.array(data)
        array = np.sort(array)[::-1]
        total = np.sum(array[::2])
        try:
            value = float(result)
            if not np.allclose(value, total):
                raise
            print('Result OK.')
        except:
            s = '{}'.format(total)
            if result.find(s) >= 0:
                print('Result OK.')
            else:
                print('Expected: {}'.format(total))
                print('Result:')
                print(result)


    def test_matrix(self, test):
        print()
        print('='*72)
        print(test)
        print('-'*72)
        sample_filename = 'matrix.txt'
        result = subprocess.check_output(
            [self.python,
             os.path.join(self.path, test + '.py'),
             sample_filename],
            stderr = subprocess.STDOUT,
            universal_newlines = True)
        data = []
        with open(sample_filename, 'rt') as f:
            lines = list(f)
        m1 = []
        m2 = []
        n = len(lines[0].split())
        for i in range(n):
            m1 += [[float(v) for v in lines[i].split()]]
        for i in range(n):
            m2 += [[float(v) for v in lines[i+n+1].split()]]
        m1 = np.array(m1)
        m2 = np.array(m2)
        m = np.dot(m1,m2)
        d = np.linalg.det(m)
        l = ''
        for v in m:
            s = ['{:15.8e}'.format(x) for x in v]
            l += '   '.join(s) + '\n'
        l += '\n'
        l += str(d) + '\n'

        try:
            r = result.splitlines()
            mx = []
            for i in range(n):
                mx += [[float(v) for v in r[i].split()]]
            mx = np.array(mx)
            dx = float(r[n+1])
            assert np.allclose(m, mx)
            assert np.allclose(d, dx)
            print('Result OK.')
        except:
            print('Expected:\n{}'.format(l))
            print('Result:')
            print(result)


    def test_letter_graph(self, test):
        print()
        print('='*72)
        print(test)
        print('-'*72)
        path = 'submissions'
        sample_filename = 'text.txt'
        with open(sample_filename) as f:
            data = f.read()
        count = collections.Counter(data.upper())
        letters = string.ascii_uppercase
        letter_count = [count[c] for c in letters]
        x = np.arange(len(letters))

        xkcd = True
        if xkcd:
            usetex = mpl.rcParams['text.usetex']
            mpl.rcParams['text.usetex'] = False
            manager = plt.xkcd()
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.set_xlabel('letter')
        ax.set_ylabel('count')
        ax.set_xlim([0,len(letters)])
        width = 0.8
        ax.bar(x, letter_count, width=width, color = 'g')
        ax.set_xticks(x+width/2)
        ax.set_xticklabels(letters)
        f.tight_layout()
        if xkcd:
            mpl.rcParams.update(manager._rcparams)
            mpl.rcParams['text.usetex'] = usetex
        plt.show(block = False)

        p = subprocess.Popen(
            [self.python,
             os.path.join(self.path, test + '.py'),
             sample_filename,
             ],
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            universal_newlines = True,
            )
        plt.show()
        try:
            print(p.communicate(input='\n', timeout=1))
        except:
            print('Process did not finish properly')
        p.poll()
        if p.returncode is None:
            p.terminate()

    def test_rubik_graph(self, test):
        print()
        print('='*72)
        print(test)
        print('-'*72)
        path = 'submissions'
        sample_transform = 'bdullrf'

        Cube(plot = 'draw', s=sample_transform)
        plt.show(block = False)
        p = subprocess.Popen(
            [self.python,
             os.path.join(self.path, test + '.py'),
             sample_transform,
             ],
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            universal_newlines = True,
            )
        plt.show()
        try:
            print(p.communicate(input='\n', timeout=1))
        except:
            print('Process did not finish properly')
        p.poll()
        if p.returncode is None:
            p.terminate()

    functions = {
        '1a' : test_replace,
        '1b' : test_ion,
        '2a' : test_vector,
        '2b' : test_matrix,
        '3a' : test_letter_graph,
        '3b' : test_rubik_graph,
        }

class Nucleus(object):
    elements = np.array(['H','He','Li','Be','B','C','N','O','F','Ne',
                         'Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca'])
    def __init__(self, s):
        i = 0
        while s[i] in string.ascii_letters:
            i += 1
        A = int(s[i:])
        Z = np.argwhere(s[:i] == self.elements)[0][0]
        self.A = A
        self.Z = Z + 1
    def __str__(self):
        return(self.elements[self.Z - 1] + str(self.A))
    def __add__(self, other):
        x = copy.copy(self)
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        x.A += other.A
        x.Z += other.Z
        return x
    __repr__ = __str__

class Piece(object):
    # down, up, front, back, left, and right
    colors = np.array(['#ffd500','#ffffff','#b71234','#ff5800','#009b48','#0046ad','#000000'])
    width = 0.9
    mutations = {
        'f': [4,5,2,3,1,0],
        'b': [5,4,2,3,0,1],
        'd': [0,1,5,4,2,3],
        'u': [0,1,4,5,3,2],
        'l': [3,2,0,1,4,5],
        'r': [2,3,1,0,4,5],
        }
    zdirs = np.array(['z','z','y','y','x','x']) # could use zdirs = 'zzyyxx'
    offsets = np.array([0,1,0,1,0,1])
    maps = np.array([[0,1,2]]*2 + [[0,2,1]]*2 + [[1,2,0]]*2)
    def __init__(self, location = None):
        self.faces = np.arange(6)
        if location is not None:
            self.set_interior(location)
    def set_interior(self, location):
        location = location[::-1]
        location = np.dstack([location, location]).flatten()
        pos = location != np.array([-1,1,-1,1,-1,1])
        self.faces[pos] = 6
    def transform(self, t):
        self.faces = self.faces[self.mutations[t]]
    def draw_face(self, ax, x,y,z, face):
        w = self.width
        xyz = np.array([x,y,z])-w/2
        xyz = xyz[self.maps[face]]
        xyz[-1] += self.offsets[face]*w
        r = Rectangle(xyz[0:2],w,w,
                      color=self.colors[self.faces[face]],
                      ec = 'k')
        ax.add_patch(r)
        art3d.pathpatch_2d_to_3d(r,
                                 z = xyz[2],
                                 zdir=self.zdirs[face])
    def draw_faces(self, x,y,z, ax = None):
        for i in np.arange(6):
            self.draw_face(ax, x,y,z,i)

    def draw_bar3d(self, x,y,z, ax = None):
        w = self.width
        x -= w/2; y -= w/2; z-= w/2
        c = self.colors[self.faces]
        ax.bar3d(x,y,z, w,w,w,
                 color = c,
                 edgecolors = 'k')
    def draw(self, *args, shade = False, **kwargs):
        if shade:
            self.draw_bar3d(*args, **kwargs)
        else:
            self.draw_faces(*args, **kwargs)

class Cube(object):
    slices = {
        'd': np.s_[::-1,:,0],
        'l': np.s_[0,::-1,:],
        'f': np.s_[:,0,:],
        'u': np.s_[:,:,-1],
        'r': np.s_[-1,:,:],
        'b': np.s_[:,-1,::-1],
        }
    def __init__(self, s = None, plot = False, **kwargs):
        self.pieces = np.ndarray((3,3,3), dtype = np.object)
        it = np.nditer([self.pieces], flags=['multi_index', 'refs_ok'], op_flags = ['writeonly'])
        while not it.finished:
            coord = np.array(it.multi_index)-1
            it[0] = Piece(coord)
            it.iternext()
        self.pieces[1,1,1] = None
        if s is not None:
            self.transform(s)
        if plot is not False:
            self.plot(show = plot, **kwargs)
    def transform(self, t = ''):
        if len(t) == 0:
            return
        if len(t) > 1:
            for c in t:
                self.transform(c)
            return
        i = self.slices[t]
        for p in self.pieces[i].flat:
            p.transform(t)
        self.pieces[i] = np.rot90(self.pieces[i])
    def plot(self, show = True, debug = None, **kwargs):
        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')
        ax.set_xlim3d(-2, 2)
        ax.set_ylim3d(-2, 2)
        ax.set_zlim3d(-2, 2)
        res = mpl.rcParams.copy()
        if debug is True:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        else:
            plt.xkcd()
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_zticks([])
            ax._axis3don = False
        it = np.nditer([self.pieces], flags=['multi_index', 'refs_ok'])
        while not it.finished:
            cube = it[0][()]
            if cube is not None:
                coord = np.array(it.multi_index)-1
                cube.draw(*coord, ax = ax, **kwargs)
            it.iternext()
        f.tight_layout()
        if show is True:
            plt.show()
        mpl.rcParams.update(res)

class Comparer(object):
    def __init__(self):
        print('Statistics on similarity of submissions:')
        print('='*40)
        submissions_dir = '/home/alex/Class/M40001-python-2014/submisions/'
        d = difflib.SequenceMatcher()
        # cases = list(itertools.starmap(operator.concat,itertools.product(['1','2','3'], ['a', 'b'])))
        cases = [str(i+1) + j for i in range(3) for j in string.ascii_lowercase[0:2]]
        for c in cases:
            samples = sorted(glob.glob(os.path.join(submissions_dir, '*'+c+'.py')))
            for i, x1 in enumerate(samples):
                s = '{:<20s}: '.format(x1.split('/')[-1].split('.')[0])
                with open(x1, 'rt') as f:
                    d.set_seq1(f.read())
                for x2 in samples[:i]:
                    if x1 == x2:
                        s += ' {:>5s}'.format('---')
                    else:
                        with open(x2, 'rt') as f:
                            d.set_seq2(f.read())
                        s += ' {:5.1f}'.format(d.ratio() * 100)
                print(s)
            print()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        Comparer()
    else:
        name = sys.argv[1]
        Tester(name)

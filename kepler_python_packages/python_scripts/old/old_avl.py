"""
Provide sorted, indexed dictionary based on AVL Tree.
"""

# TODO - check itertems/items, iterkeys/keys

import operator
import collections

from utils import isinslice, bool2sign
from copy import deepcopy

class AVL_Leaf(object):
    """
    Leaf Object that cimbines administrative information.

    There is a set of iterator routines that are built on the ietartor
    protocal using 'yield' rather than the set of tree walk routines
    that maintaina pointer list and which are implemented on the node
    level (in the node class).

    It also defines a set of 'convenince methods' that map to the
    corresponding methods of the contained node object (hence do not
    provide their own __doc__ string).
    """
    __slots__ = ("node", "count", "depth")
    def __init__(self, node = None):
        self.node = node
        if node == None:
            self.count = 0
            self.depth = 0
        else:
            self.count = node.count()
            self.depth = node.depth()

    @classmethod
    def new(cls, key, value):
        """
        Create and retund new AVL_Leaf object.
        """
        return cls(AVL_Node(key, value))

    def empty(self):
        """
        Tell if leaf is used.
        """
        return self.node == None

    def iternodes(self, reverse = False):
        """
        Return node iterator.
        """
        if self.empty():
            raise StopIteration
        for node in self.node.leaf[reverse].iternodes(reverse):
            yield node
        yield self.node
        for node in self.node.leaf[not reverse].iternodes(reverse):
            yield node

    def iternodes_n(self, step = 1, offset = 0):
        """
        Return node step iterator. (Version 2)

        'offset' is relative to left.
        """
        def leaf0():
            if (((offset+step-1) % step) + 1) + self.node.leaf[0].count >= step:
                for node in self.node.leaf[0].iternodes_n(step, offset):
                    yield node
        def leafm():
            if (self.node.leaf[0].count + offset) % step == 0:
                yield self.node
        def leaf1():
            if (((offset + self.node.leaf[0].count + step) % step) + 1) + self.node.leaf[1].count >= step:
                for node in self.node.leaf[1].iternodes_n(step, offset + self.node.leaf[0].count + 1):
                    yield node

        if self.empty():
            raise StopIteration
        leafs = [leaf0, leafm, leaf1]
        if step < 0:
            leafs.reverse()
        for leaf in leafs:
            for node in leaf():
                yield node

    def iternodes_slice(self, slice, offset = 0):
        """
        Return node slice iterator.

        'offset' is relative to left.
        """
        def leaf0():
            if slice.step > 0:
                if (offset >= slice.stop) or (o0 - 1 < slice.start):
                    raise StopIteration()
            else:
                if (offset > slice.start) or (o0 - 1 <= slice.stop):
                    raise StopIteration()
            if (((offset + slice.step - 1 - slice.start) % slice.step) + 1) + self.node.leaf[0].count >= slice.step:
                for node in self.node.leaf[0].iternodes_slice(slice, offset):
                    yield node
        def leafm():
            if slice.step > 0:
                if (o0 >= slice.stop) or (o0 < slice.start):
                    raise StopIteration()
            else:
                if (o0 > slice.start) or (o0 <= slice.stop):
                    raise StopIteration()
            if (o0 - slice.start) % slice.step == 0:
                yield self.node
        def leaf1():
            o1 = offset + self.count
            if slice.step > 0:
                if (o0 + 1 >= slice.stop) or (o1 < slice.start):
                    raise StopIteration()
            else:
                if (o0 + 1 > slice.start) or (o1 <= slice.stop):
                    raise StopIteration()
            if (((o0 + slice.step - slice.start) % slice.step) + 1) + self.node.leaf[1].count >= slice.step:
                for node in self.node.leaf[1].iternodes_slice(slice, o0 + 1):
                    yield node

        if self.empty():
            raise StopIteration
        leafs = [leaf0, leafm, leaf1]
        if self.node is not None:
            o0 = offset + self.node.leaf[0].count
        if slice.step < 0:
            leafs.reverse()
        for leaf in leafs:
            for node in leaf():
                yield node

    def iternodes_list(self, list, offset = 0):
        """
        Return node iterator for given list of indices.

        'offset' is relative to left.

        'list' is *reverse* index list of items to return

        negative index (index relative to end of list)
          is not allowed at this time.

        The list is consumed:
          Entries are removed from when the node is returned.

        Items are taken from the *end* of the list -
          hence, to receive items in order of list you need to call the
          routine with a reversed list, list[::-1].

          This may also make a copy of the list (in Python 2.7.2) and
          the original copy of the list is preserved, in contrast to
          a call like

          >>> MyList = [1,2,3]
          >>> Mylist.reverse()
          >>> MyLeaf.iternodes_list(MyList)

          In case later versions return a view or an iterable
          for [::-1], the behavior may be different.
        """
        if self.node is None:
            raise StopIteration()
        while len(list) > 0 and offset <= list[-1] < offset + self.count: 
            if list[-1] == offset + self.node.leaf[0].count:
                list.pop()
                yield self.node
            for node in self.node.leaf[0].iternodes_list(list, offset = offset):
                yield node
            for node in self.node.leaf[1].iternodes_list(list, offset = offset + self.node.leaf[0].count + 1):
                yield node

    # convenience routines
    # see documentation below in AVL_Node
    def rebalance(self):
        return self.node.rebalance()
    def balance(self):
        return self.node.balance()
    def insert(self, *args, **kwargs):
        return self.node.insert(*args, **kwargs)
    def insert_index(self, *args, **kwargs):
        return self.node.insert_index(*args, **kwargs)
    def pop(self, *args, **kwargs):
        return self.node.pop(*args, **kwargs)
    def delete(self, *args, **kwargs):
        return self.node.delete(*args, **kwargs)
    def delete_index(self, *args, **kwargs):
        return self.node.delete_index(*args, **kwargs)
    def rotation(self, *args, **kwargs):
        return self.node.rotation(*args, **kwargs)
    def tree(self, *args, **kwargs):
        return self.node.tree(*args, **kwargs)
    def find(self, *args, **kwargs):
        return self.node.find(*args, **kwargs)
    def find_index(self, *args, **kwargs):
        return self.node.find_index(*args, **kwargs)
    def find_end(self, *args, **kwargs):
        return self.node.find_end(*args, **kwargs)
    def find_pos(self, *args, **kwargs):
        return self.node.find_pos(*args, **kwargs)
    def find_value(self, *args, **kwargs):
        return self.node.find_value(*args, **kwargs)
    def find_value_pos(self, *args, **kwargs):
        return self.node.find_value_pos(*args, **kwargs)
    def get_nodes(self, *args, **kwargs):
        return self.node.get_nodes(*args, **kwargs)
    def copy(self, *args, **kwargs):
        return self.node.copy(*args, **kwargs)
    def pop_deep(self):
        return self.node.pop_deep()

    ############################################################
    # ATTIC
    ############################################################

    # def find_index_slice(self, *args, **kwargs):
    #     return self.node.find_index_slice(*args, **kwargs)

    ############################################################

class AVL_Node(object):
    """
    AVL Tree node obejct.
    """
    __slots__ = ("value","key","leaf")
    def __init__(self,
                 key,
                 value = None):
        self.value = value
        self.key = key
        self.new_leaf()

    def new_leaf(self):
        self.leaf = [AVL_Leaf(),AVL_Leaf()]

    def count(self):
        return 1 + self.leaf[0].count + self.leaf[1].count

    def depth(self):
        return 1 + max(self.leaf[0].depth, self.leaf[1].depth)

    def balance(self):
        return self.leaf[0].depth - self.leaf[1].depth

    @staticmethod
    def first_walk(stack, right = False):
        """
        Set up stack to leftmost or rightmost node.
        """
        leaf = stack[-1].leaf[right]
        while not leaf.empty():
            stack.append(leaf.node)
            leaf = leaf.node.leaf[right]

    @classmethod
    def step_walk(cls, stack, right = False):
        """
        Make one step coming from [right] direction using the provided stack.
        """
        leaf = stack[-1].leaf[not right]
        if not leaf.empty():
            stack.append(leaf.node)
            cls.first_walk(stack, right)
        else:
            node = stack.pop()
            while (len(stack) > 0) and (stack[-1].leaf[not right].node == node):
                node = stack.pop()

    @staticmethod
    def step_walk_n(stack, n = 0):
        """
        Make n steps, positive for rightward, negative for leftward, the provided stack.
        """
        right = n > 0
        leaf = stack[-1].leaf[right]
        while abs(n) > leaf.count:
            sig = bool2sign(right)
            n -= leaf.count * sig
            node = stack.pop()
            if len(stack) == 0:
                return
            leaf = stack[-1].leaf[right]
            if node == leaf.node:
                n += leaf.count * sig
            else:
                n -= sig
            right = n > 0
            leaf = stack[-1].leaf[right]
        while n != 0:
            stack.append(leaf.node)
            n -= (stack[-1].leaf[not right].count + 1) * bool2sign(right)
            if n != 0:
                right = n > 0
                leaf = stack[-1].leaf[right]
        # while n != 0:
        #     stack.append(stack[-1].leaf[n > 0].node)
        #     n -= (stack[-1].leaf[n < 0].count + 1)  * cmp(n, 0)

    @staticmethod
    def index_walk(stack, index):
        """
        Set up stack to element with index.
        """
        node = stack[-1]
        pos = node.leaf[0].count
        while index != pos:
            right = index > pos
            node = node.leaf[right].node
            stack.append(node)
            pos += bool2sign(right)*(node.leaf[not right].count + 1)

    @staticmethod
    def key_walk(stack, key):
        """
        Set up stack to element with key.
        """
        node = stack[-1]
        while node.key != key:
            right = key > node.key
            node = node.leaf[right].node
            if node == None:
                break
            stack.append(node)

    # This seems to be efficient rarely
    @classmethod
    def step_walk_key(cls, stack, key):
        """
        Steps to key provided stack.
        """
        while len(stack) > 1:
            if stack[-1].key == key:
                break
            if cmp(stack[-2].key, key) * cmp(stack[-1].key, key) < 0:
                break
            stack.pop()
        cls.key_walk(stack, key)


    def find(self, key):
        if self.key == key:
            return self
        else:
            right = key > self.key
            if self.leaf[right].empty():
                return None
            else:
                return self.leaf[right].find(key)

    def insert(self, key, value):
        if key == self.key:
            self.value = value
        else:
            right = key > self.key
            if self.leaf[right].empty():
                self.leaf[right] = AVL_Leaf.new(key, value)
            else:
                self.leaf[right].insert(key, value)
                self.leaf[right] = self.leaf[right].rebalance()

    def delete(self, key):
        if key == self.key:
            if self.count() == 1:
                return self, AVL_Leaf()
            if self.balance() == 0:
                right = self.leaf[0].count <= self.leaf[1].count
            else:
                right = self.balance() < 0
            node, leaf = self.leaf[right].pop(not right)
            # node.leaf[right] = leaf
            # node.leaf[not right] = self.leaf[not right]
            node.leaf[::bool2sign(right)] = [self.leaf[not right],leaf]
            self.new_leaf()
            return self, node.rebalance()
        right = key > self.key
        if self.leaf[right].empty():
            return None, AVL_Leaf(self)
        else:
            node, self.leaf[right] = self.leaf[right].delete(key)
            return node, self.rebalance()

    def pop(self, right):
        if self.leaf[right].empty():
            return self, self.leaf[not right]
        else:
            node, self.leaf[right] = self.leaf[right].pop(right)
            return node, self.rebalance()

    def pop_deep(self):
        if self.count() == 1:
            return self, AVL_Leaf()
        if self.balance() == 0:
            right = self.leaf[0].count <= self.leaf[1].count
        else:
            right = self.balance() < 0
        node, self.leaf[right] = self.leaf[right].pop_deep()
        return node, self.rebalance()

    def insert_index(self, index, value, key = None, offset = 0):
        """
        Insert node by index.

        WARNING:
          This should only be used if no soring by key is desired,
          that is, when implementing an *ordered* dictionary
          rather than a *sorted* dictionary.

          In that case the interface interface object should use
          an iterator to locate a key (which is inefficient).
        """
        pos = offset + self.leaf[0].count
        right = index > pos
        if right:
            offset = pos + 1
        if self.leaf[right].empty():
            self.leaf[right] = AVL_Leaf.new(key, value)
        else:
            self.leaf[right].insert_index(index, value, key = key, offset = offset)
            self.leaf[right] = self.leaf[right].rebalance()

    def delete_index(self, index, offset = 0):
        """
        Hated to copy most of the 'delete' code.
        """
        pos = offset + self.leaf[0].count
        if index == pos:
            if self.count() == 1:
                return self, AVL_Leaf()
            if self.balance() == 0:
                right = self.leaf[0].count <= self.leaf[1].count
            else:
                right = self.balance() < 0
            node, leaf = self.leaf[right].pop(not right)
            node.leaf[::bool2sign(right)] = [self.leaf[not right],leaf]
            self.new_leaf()
            return self, node.rebalance()
        right = index > pos
        if right:
            offset += self.leaf[0].count + 1
        node, self.leaf[right] = self.leaf[right].node.delete_index(index, offset)
        return node, self.rebalance()
        # raise NotImplementedError("Use delete(find_index(index)) instead.")


    def rotation(self, right):
        """
        Rotate tree around self.
        """
        node = self.leaf[not right].node
        self.leaf[not right] = node.leaf[right]
        node.leaf[right] = AVL_Leaf(self)
        return AVL_Leaf(node)

    def rebalance(self):
        if -1 <= self.balance() <= +1:
            return AVL_Leaf(self)
        right = self.balance() == -2
        if self.leaf[right].balance()*bool2sign(right) > 0:
            self.leaf[right] = self.leaf[right].rotation(right)
        return self.rotation(not right)

    def find_index(self, index, offset = 0):
        pos = offset + self.leaf[0].count
        if index == pos:
            return self
        # if index < pos:
        #     return self.leaf[0].node.find_index(index, offset)
        # return self.leaf[1].node.find_index(index, offset + self.leaf[0].count + 1)
        right = index > pos
        if right:
            offset += self.leaf[0].count + 1
        return self.leaf[right].node.find_index(index, offset)

    def find_end(self, right = True):
        if self.leaf[right].empty():
            return self
        return self.leaf[right].find_end(right)

    def find_pos(self, key):
        if self.key == key:
            return self.leaf[0].count
        if key < self.key:
            return self.leaf[0].find_pos(key)
        return self.leaf[0].count + 1 + self.leaf[1].find_pos(key)
        # right = key > self.key
        # pos = (self.leaf[0].count + 1) if right else 0
        # return pos + self.leaf[right].find_pos(key)

    def find_value(self, value):
        if self.value == value:
            return self
        for right in (False, True):
            if not self.leaf[right].empty():
                node = self.leaf[right].find_value(value)
                if node is not None:
                    return node
        return None

    def get_nodes(self, nodes = None):
        if not self.leaf[0].empty():
            self.leaf[0].get_nodes(nodes)
        nodes.append(self)
        if not self.leaf[1].empty():
            self.leaf[1].get_nodes(nodes)

    def find_value_pos(self, value):
        if self.value == value:
            return self.leaf[0].count
        for right in (False, True):
            if not self.leaf[right].empty():
                pos = self.leaf[right].find_value_pos(value)
                if pos >= 0:
                    if right:
                        return pos + self.leaf[0].count + 1
                    return pos
        return -1

    def copy(self, leaf, deepcopy = False):
        """
        shallow copy
        """
        if deepcopy:
            leaf.node = self.__class__(
                deepcopy(self.key),
                deepcopy(self.value))
        else:
            leaf.node = self.__class__(
                self.key,
                self.value)
        for right in [False, True]:
            if self.leaf[right].empty():
                leaf.node.leaf[right] = AVL_Leaf()
            else:
                self.leaf[right].copy(leaf.node.leaf[right])

    def tree(self, s):
        tree = ''
        index = 0
        s1 = "{:>2}[{:>2}]({}): {}".format(
            self.key,
            self.balance(),
            self.count(),
            self.value)
        s2 = "".ljust(4)
        if not self.leaf[1].empty():
            tree += self.leaf[1].tree(s + s2)
        tree += s + s1 + "\n"
        if not self.leaf[0].empty():
            tree += self.leaf[0].tree(s + s2)
        return tree

    ############################################################
    # ATTIC
    ############################################################

    # the following routine is probably superfluous
    #
    # use the "walk" set of methods instead for slice implementation
    # this would also be more compatible with the Py3 concept
    #   of using iterators
    # def find_index_slice(self, index, result = None, offset = 0):
    #     if not self.leaf[0].empty():
    #         self.leaf[0].find_index_slice(index, result, offset)
    #     pos = offset + self.leaf[0].count
    #     if isinslice(pos, index):
    #         if index.indices(0)[2] > 0:
    #             result.append(self)
    #         else:
    #             result.insert(0,self)
    #     if not self.leaf[1].empty():
    #         self.leaf[1].find_index_slice(index,result,offset + self.leaf[0].count + 1)

    ############################################################




class SortedDict(object):
    """
    Interface for AVL Tree sorted dictionary that also is indexable.

    In contrast to ordinary dictional, the keys need to be comparable
    by inequality relations.

    Currently syntacx for '[]' interace is unfortunate.  If the passed
    value us a slice it will trigger slice indexing, otherwise the
    argument is interpreted as a key.  We provide the '()' call
    interface for single indices.  The () cannot be used for
    assignment, however.  With the '[]' we get implicit conversion to
    a slice on call for free, however.

    If a list was passed, should it be a list of keys or list of indices?
    (not implemeneted)

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize AVL tree (from iterable, if provided)

        Use kwarg 'type' or first or last non-kw arg to specify type.

        """
        self.root = AVL_Leaf()
        self.type = kwargs.pop('type',None)
        if len(args) > 0:
            if isinstance(args[0], type):
                self.type = args[0]
                args = args[1:]
            elif isinstance(args[-1], type):
                self.type = args[-1]
                args = args[:-1]
        self.update(*args, **kwargs)

    # dictionary interface

    def update(self, *args, **kwargs):
        """
        Insert/update elements.
        """
        if len(args) == 1:
            for k,v in args[0]:
                self.insert(k, v)
        elif len(args) == 2:
            for k,v in zip(args[0],args[1]):
                self.insert(k, v)
        elif len(args) > 0:
            raise ValueError("Invalid argument list.")
        for k,v in kwargs.items():
            self.insert(k, v)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        dictionary = cls()
        for key in iterable:
            dictionary[key] = value
        return dictionary

    def setdefault(self, key, value = None):
        if self.root.empty():
            self.root = self.root.new(key, value)
        else:
            item = self.root.get_key(key)
            if item == None:
                self.root.insert(key, value)
            else:
                value = item
        return value

    def insert(self, key, value = None):
        if self.root.empty():
            self.root = self.root.new(key, value)
        else:
            self.root.insert(key, value)
            self.root = self.root.rebalance()

    def get(self, key, value = None):
        if self.root.empty():
            return value
        node = self.root.find(key)
        if node == None:
            return value
        else:
            return node.value

    def pop(self, key, value = None):
        if self.root.empty():
            return value
        item = self.delete(key)
        if item == None:
            return value
        return item

    def popitem(self):
        if self.root.empty():
            return value
        node, self.root = self.root.pop_deep()
        if node == None:
            return None
        else:
            return node.value

    def clear(self):
        self.root = AVL_Leaf()

    def copy(self):
        root = AVL_Leaf()
        self.root.copy(root)
        duplicate = self.__class__()
        duplicate.root = root
        return duplicate

    def delete(self, item):
        if issubclass(type(item), int):
            key, value = self.delete_index(item)
        else:
            if self.type is not None:
                if not issubclass(type(item), self.type):
                    raise TypeError("invalid key type")
            key, value = self.delete_key(item)
        return value

    # specific routines

    def delete_key(self, key):
        # this may need to be generalized to retudn node or key as well
        if self.root.empty():
            node = None
        else:
            node, self.root = self.root.delete(key)
        if node == None:
            return None, None
        else:
            return node.value, node.value

    def delete_index(self, index):
        # this may need to be generalized to retudn node or key as well
        if index < 0:
            index += self.root.count
        if not 0 <= index < self.root.count:
            raise IndexError("index out of range")
        node, self.root = self.root.delete_index(index)
        return node.key, node.value

    def iternodes_indexlist(self, indexlist):
        for index in indexlist:
            yield self.node_index(index)

    def iternodes_keylist(self, keylist):
        if len(keylist) <= 1:
            for key in keylist:
                yield self.node_key(key)
        else:
            # this may only be efficient for dense key lists
            if self.root.empty():
                raise StopIteration("empty dictionary")
            stack = [self.root.node]
            self.root.node.key_walk(stack, keylist[0])
            yield stack[-1]
            for key in keylist[1:]:
                self.root.node.step_walk_key(stack, key)
                if stack[-1].key == key:
                    yield stack[-1]
                else:
                    yield None

    def values_indexlist(self, *args, **kwargs):
        """
        Return list of values for given list of indices.
        """
        return [node.value for node in self.iternodes_indexlist(*args, **kwargs)]
    def values_keylist(self, *args, **kwargs):
        """
        Return list of values for given list of keys
        """
        return [node.value for node in self.iternodes_keylist(*args, **kwargs)]
    # etc.


    # another test
    def iternodes_step(self, step, right = False):
        for node in self.root.iternodes_n(step, (-self.root.count + 1)*int(right)):
            yield node
    def iternodes_slice(self, slc):
        for node in self.root.iternodes_slice(slice(*slc.indices(self.root.count)),0):
            yield node

    # good routines
    def iternodes_indexslice(self, *args):
        if self.root.empty():
            raise StopIteration("empty dictionary")
        if len(args) == 1 and issubclass(type(args[0]),slice):
            s = args[0]
        else:
            if len(args) == 0:
                args = [None]
            s = slice(*args)
        start, stop, step = s.indices(self.root.count)
        reverse = step < 1
        if operator.xor(stop < start, reverse):
            raise StopIteration("empty range")
        stack = [self.root.node]
        self.root.node.index_walk(stack, start)

        # the threshold value to be adjusted
        if abs(step) > 2:
            for i in range(start, stop, step):
                if not (i == start):
                    self.root.node.step_walk_n(stack, step)
                yield stack[-1]
        else:
            for i in range(start, stop, cmp(step, 0)):
                if (i - start) % step == 0:
                    yield stack[-1]
                self.root.node.step_walk(stack, reverse)
                if len(stack) == 0:
                    break

    def iteritems_indexslice(self, *args):
        for node in self.iternodes_indexslice(*args):
            yield node.key, node.value

    def itervalues_indexslice(self, *args):
        for node in self.iternodes_indexslice(*args):
            yield node.value

    def iterkeys_indexslice(self, *args):
        for node in self.iternodes_indexslice(*args):
            yield node.key

    def iternodes_keyslice(self, *args):
        if self.root.empty():
            raise StopIteration("empty dictionary")
        if len(args) == 1 and issubclass(type(args[0]),slice):
            s = args[0]
        else:
            if len(args) == 0:
                args = [None]
            s = slice(*args)
        start, stop, step = s.start, s.stop, s.step
        if step == 0:
            raise ValueError("step argument must not be zero")
        elif step == None:
            step = 1
        reverse = step < 1
        stack = [self.root.node]
        if start == None:
            self.root.node.first_walk(stack, reverse)
        else:
            self.root.node.key_walk(stack, start)
            if (stack[-1].key != start) and \
                    operator.xor(stack[-1].key < start, reverse):
                self.root.node.step_walk(stack, reverse)
                if len(stack) == 0:
                    raise StopIteration()
        if abs(step) == 1:
            while operator.xor(stack[-1].key < stop, reverse) or (stop == None):
                yield stack[-1]
                self.root.node.step_walk(stack, reverse)
                if len(stack) == 0:
                    break

        # threshold to adjust
        elif abs(step) <= 2:
            i = 0
            while operator.xor(stack[-1].key < stop, reverse) or (stop == None):
                if i % step == 0:
                    yield stack[-1]
                self.root.node.step_walk(stack, reverse)
                if len(stack) == 0:
                    break
                i += 1
        else:
            while operator.xor(stack[-1].key < stop, reverse) or (stop == None):
                yield stack[-1]
                self.root.node.step_walk_n(stack, step)
                if len(stack) == 0:
                    break

    def iteritems_keyslice(self, *args):
        for node in self.iternodes_keyslice(*args):
            yield node.key, node.value

    def itervalues_keyslice(self, *args):
        for node in self.iternodes_keyslice(*args):
            yield node.value

    def iterkeys_keyslice(self, *args):
        for node in self.iternodes_keyslice(*args):
            yield node.key

    # superfluous
    def values_indexslice(self, *args):
        return [value for value in self.itervalues_indexslice(*args)]
    def values_keyslice(self, *args):
        return [value for value in self.itervalues_keyslice(*args)]
    def keys_indexslice(self, *args):
        return [key for key in self.iterkeys_indexslice(*args)]
    def keys_keyslice(self, *args):
        return [key for key in self.iterkeys_keyslice(*args)]
    def items_indexslice(self, *args):
        return [item for item in self.iteritems_indexslice(*args)]
    def items_keyslice(self, *args):
        return [item for item in self.iteritems_keyslice(*args)]
    def nodes_indexslice(self, *args):
        return [node for node in self.iternodes_indexslice(*args)]
    def nodes_keyslice(self, *args):
        return [node for node in self.iternodes_keyslice(*args)]

    # code dupcation from slice version, but more streamlined
    def iternodes(self, reverse = False):
        """
        Efficient iterator over all nodes.
        """
        # if not self.root.empty():
        #     stack = [self.root.node]
        #     self.root.node.first_walk(stack, reverse)
        #     while len(stack) > 0:
        #         yield stack[-1]
        #         self.root.node.step_walk(stack, reverse)
        for node in self.root.iternodes(reverse):
            yield node

    def iteritems(self, *args, **kwargs):
        for node in self.iternodes( *args, **kwargs):
            yield node.key, node.value

    def iterkeys(self, *args, **kwargs):
        for node in self.iternodes( *args, **kwargs):
            yield node.key

    def itervalues(self, *args, **kwargs):
        for node in self.iternodes( *args, **kwargs):
            yield node.value


    # INEFFICIENT
    def nodes(self):
        nodes = []
        if not self.root.empty():
            self.root.get_nodes(nodes)
        return nodes

    # INEFFICIENT
    def items(self):
        return [(node.key, node.value) for node in self.nodes()]

    # INEFFICIENT
    def keys(self):
        return [node.key for node in self.nodes()]

    # INEFFICIENT
    def values(self):
        return [node.value for node in self.nodes()]

    def node_index(self, index):
        """
        return node for given index
        """
        if self.root.empty():
            return None
        if index < 0:
            index += self.root.count
        if not 0 <= index < self.root.count:
            raise IndexError("index out of range")
        return self.root.find_index(index)

    def key_index(self, index):
        """
        return key for given index
        """
        node = self.root.find_index(index)
        if node == None:
            return None
        return node.key

    def node_key(self, key):
        if self.root.empty():
            return None
        return self.root.find(key)

    def index_from_key(self, key):
        if self.root.empty():
            return -1
        return self.root.find_pos(key)

    def index_from_value(self, value):
        if self.root.empty():
            return -1
        return self.root.find_value_pos(value)

    def key_from_value(self, value):
        if self.root.empty():
            return None
        node = self.root.find(key)
        if node == None:
            return None
        else:
            return node.key

    def max_key(self):
        if self.root.empty():
            return None
        return self.root.find_end(right = True)

    def min_key(self):
        if self.root.empty():
            return None
        return self.root.find_end(right = False)

    def has_key(self, key):
        if self.root.empty():
            return False
        return self.root.find(key) is not None

    __contains__ = has_key

    def has_value(self, value):
        if self.root.empty():
            return False
        return self.root.find_value(value) is not None

    # change?
    def __call__(self, index):
        return self.node_index(index).value

    def __len__(self):
        return self.root.count
    def __size__(self):
        return self.root.depth

    def _get_slice(self, s):
        assert isinstance(s,slice), "need slice type"
        args = (s.start, s.stop, s.step)
        iskey = False
        isint = False
        for arg in args:
            if issubclass(type(arg), int):
                isint = True
            if self.type is not None:
                if issubclass(type(arg), self.type):
                    iskey = True
        if (self.type == None) and (isint == False):
            iskey == True
        if iskey == True:
            return self.values_keyslice(s)
        return self.values_indexslice(s)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._get_slice(key)
        # not sure this syntax should be allowed in the first place...
        if issubclass(type(key),collections.Iterable):
            return [self.__getitem__(xkey) for xkey in key]
        if self.type is not None:
            if not issubclass(type(key),self.type):
                return self.node_index(key).value
        return self.node_key(key).value

    def _set_indexslice(self, s, values):
        for node, value in zip(iternodes_indexslice(s), values):
            node.value = value

    def _set_keyslice(self, s, values):
        for node, value in zip(iternodes_keyslice(s), values):
            node.value = value

    def _set_slice(self, s, value):
        assert issubclass(type(s),slice), "need slice type"
        args = (s.start, s.stop, s.step)
        iskey = False
        isint = False
        for arg in args:
            if issubclass(type(arg), int):
                isint = True
            if self.type is not None:
                if issubclass(type(arg), self.type):
                    iskey = True
        if (self.type == None) and (isint == False):
            iskey == True
        if iskey == True:
            self._set_keyslice(s, value)
            return
        self._set_indexslice(s, value)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self._set_slice(key, value)
            return
        # not sure this syntax should be allowed in the first place...
        if issubclass(type(key),collections.Iterable):
            for xkey, xvalue in zip(key, value):
                self.__setitem__(xkey, xvalue)
            return
        if self.type is not None:
            if not issubclass(type(key),self.type):
                self.node_index(key).value = value
                return
        self.insert(key, value)

    def _del_indexslice(self, s):
        # conservative approach due to rebalancing
        # first create list of keys, then delete them, one by one
        # we *must* not change tree while using an iterator!
        keys = self.keys_indexslice(s)
        for key in keys:
            self.delete(key)

    def _del_keyslice(self, s):
        # conservative approach due to rebalancing
        # first create list of keys, then delete them, one by one
        # we *must* not change tree while using an iterator!
        keys = self.keys_keyslice(s)
        for key in keys:
            self.delete(key)

    def _del_slice(self, s):
        assert issubclass(type(s),slice), "need slice type"
        args = (s.start, s.stop, s.step)
        iskey = False
        isint = False
        for arg in args:
            if issubclass(type(arg), int):
                isint = True
            if self.type is not None:
                if issubclass(type(arg), self.type):
                    iskey = True
        if (self.type == None) and (isint == False):
            iskey == True
        if iskey == True:
            self._del_keyslice(s)
            return
        self._del_indexslice(s)

    def __delitem__(self, key):
        if isinstance(key, slice):
            self._del_slice(key)
            return
        # not sure this syntax should be allowed in the first place...
        if issubclass(type(key),collections.Iterable):
            for xkey in key:
                self.__delitem__(xkey)
            return
        if self.type is not None:
            if not issubclass(type(key),self.type):
                self.delete_index(key)
                return
        self.delete(key)

    # list interface

    def pop_end(self, right = True):
        if self.root.empty():
            return value
        node, self.root = self.root.pop(right)
        if node == None:
            return None
        else:
            return node.value

    def popleft(self):
        return self.pop(right = False)

    def popright(self):
        return self.pop(right = True)

    def deepcopy(self):
        root = AVL_Leaf()
        self.root.copy(root, deepcopy = True)
        duplicate = self.__class__()
        duplicate.root = root
        return duplicate

    def tree(self):
        if not self.root.empty():
            return self.root.tree('')
        else:
            return "<empty>"

    def print(self):
        print(self.tree())

    def _name(self):
        s = ''
        for k,v in self.items():
            if s != '': s += ', '
            s += "{!r}: {!r}".format(k,v)
        return '{'+s+'}'

    def __str__(self):
        return self._name()

    def __repr__(self):
        return self.__class__.__name__ + "("+self._name()+")"

    def __iter__(self):
        return self.iterkeys()

    def viewitems(self):
        return SortedDictItemsView(self)

    def viewkeys(self):
        return SortedDictKeysView(self)

    def viewvalues(self):
        return SortedDictValuesView(self)

    ############################################################
    # TESTING ONLY
    ############################################################
   
    # THIS METHOD SHOULD NOT BE USED FOR SORTED DICTIONARY,
    # ONLY FOR ORDERED LIST
    def insert_index(self, index = None, value = None, key = None):
        if self.root.empty():
            self.root = self.root.new(key, value)
        else:
            if index is None:
                index = len(self)
            if index < 0:
                index += len(self)
            if index < 0:
                raise IndexError("index out of bounds")
            self.root.insert_index(index, value, key = key)
            self.root = self.root.rebalance()
    # DEFINE OTHER MTHODS LIKE append, slicing insert and replacement, ...



    ############################################################
    # ATTIC
    ############################################################

    # INEFFICIENT
    # def get_slice(self, index):
    #     nodes = []
    #     if not self.root.empty():
    #         assert isinstance(index, slice), "need slice type"
    #         self.root.find_index_slice(index,
    #                                    result = nodes)
    #     return [node.value for node in nodes]

    # INEFFICIENT
    # def set_slice(self, index, value):
    #     nodes = []
    #     if not self.root.empty():
    #         assert isinstance(index, slice), "need slice type"
    #         self.root.find_index_slice(index,
    #                                    result = nodes)
    #     if len(value) == 1:
    #         for node in nodes:
    #             node.value = value
    #     elif len(nodes) == value:
    #         for i,node in enumerate(nodes):
    #             node.value = value[i]
    #     else:
    #         raise ValueError("dimensions are not compatible")

    # def __contains__(self, key):
    #     return self.has_key(key)

    ############################################################

class SortedDictItemsView(collections.ItemsView):
    def __init__(self, parent):
        self._mapping = parent
    def __len__(self):
        return len(self._mapping)
    def __iter__(self):
        return self._mapping.iteritems()
    def __contains__(self, key):
        return key in self._mapping

class SortedDictKeysView(collections.KeysView):
    def __init__(self, parent):
        self._mapping = parent
    def __len__(self):
        return len(self._mapping)
    def __iter__(self):
        return self._mapping.iterkeys()
    def __contains__(self, key):
        return key in self._mapping

class SortedDictValuesView(collections.ValuesView):
    def __init__(self, parent):
        self._mapping = parent
    def __len__(self):
        return len(self._mapping)
    def __iter__(self):
        return self._mapping.itervalues()
    def __contains__(self, value):
        return self._mapping.has_value(value)





def test():
    import random
    a = SortedDict()
    random.seed(0)
#    for i in random.sample(range(0,999),500):
#    for i in range(7):
    for i in range(25,-1,-1):
#    for i in range(0,26):
#        print("Adding: {}".format(i))
        a.insert(2*i+1,"{:06X}".format(i))
    print("".ljust(30,'-'))
    a.print()
    print("".ljust(30,'-'))
    return a

def test2():
    import isotope
    s = isotope.SolAbu()
    a = SortedDict(s,type=isotope.Ion)
    print("".ljust(30,'-'))
    a.print()
    print("".ljust(30,'-'))
    return a


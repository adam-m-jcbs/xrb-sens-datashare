"""
Provide sorted, indexed dictionary based on AVL Tree.
"""

# TODO - check itertems/items, iterkeys/keys

import operator
import collections
import collections.abc

from utils import isinslice, bool2sign, sign
from copy import deepcopy

class AVL_Leaf(object):
    """
    Leaf Object that combines administrative information.

    There is a set of iterator routines that are built on the iterator
    protocal using 'yield' rather than the set of tree walk routines
    that maintains pointer list and which are implemented on the node
    level (in the node class).

    It also defines a set of 'convenince methods' that map to the
    corresponding methods of the contained node object (hence do not
    provide their own __doc__ string).
    """
    __slots__ = ("node", "count", "depth")
    def __init__(self, node = None):
        self.node = node
        if node is None:
            self.count = 0
            self.depth = 0
        else:
            self.count = node.count()
            self.depth = node.depth()

    @classmethod
    def new(cls, *args, **kwargs):
        """
        Create and return new AVL_Leaf object.
        """
        if len(args) == 0:
            return cls()
        if kwargs.get('key_index', False):
            return cls(AVL_key_index_Node(*args))
        if len(args) == 1:
            return cls(AVL_keyless_Node(*args))
        return cls(AVL_keyed_Node(*args))

    def empty(self):
        """
        Tell if leaf is used.
        """
        return self.node is None

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

    def __repr__(self):
        return self.__class__.__name__ + "(count: {!r}, depth: {!r}, node: {!r})".format(
            self.count, self.depth, self.node)

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

class AVL_keyless_Node(object):
    """
    AVL Tree node obejct w/o key.
    """
    __slots__ = ("value", "leaf")
    def __init__(self,
                 value = None):
        self.value = value
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

    def insert_index(self, index, value, offset = 0):
        """
        Insert node by index.

        WARNING:
          This should only be used if no sorting is desired,
          that is, when implementing an *ordered* dictionary
          rather than a *sorted* dictionary.
        """
        pos = offset + self.leaf[0].count
        right = index > pos
        if right:
            offset = pos + 1
        if self.leaf[right].empty():
            self.leaf[right] = AVL_Leaf.new(value)
        else:
            self.leaf[right].insert_index(index, value, offset = offset)
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
        copy - shallow or deep
        """
        if deepcopy:
            leaf.node = self.__class__(
                deepcopy(self.value))
        else:
            leaf.node = self.__class__(
                self.value)
        for right in [False, True]:
            if self.leaf[right].empty():
                leaf.node.leaf[right] = AVL_Leaf()
            else:
                self.leaf[right].copy(leaf.node.leaf[right])
        leaf.count = leaf.node.count()
        leaf.depth = leaf.node.depth()


    def twig(self):
        return "*[{:>2}]({}): {}".format(
            self.balance(),
            self.count(),
            self.value)

    def tree(self, s):
        tree = ''
        index = 0
        s1 = self.twig()
        s2 = "".ljust(4)
        if not self.leaf[1].empty():
            tree += self.leaf[1].tree(s + s2)
        tree += s + s1 + "\n"
        if not self.leaf[0].empty():
            tree += self.leaf[0].tree(s + s2)
        return tree

    def __repr__(self):
        return self.__class__.__name__ + "({!r}; [{!r}, {!r}])".format(
            self.value, self.leaf[0], self.leaf[1])

class AVL_keyed_Node(AVL_keyless_Node):
    """
    AVL Tree node obejct with key.

    This contains the routines that require a key
    """
    __slots__ = ("value","key","leaf")
    def __init__(self,
                 key,
                 value = None):
        self.value = value
        self.key = key
        self.new_leaf()

    def insert_index(self, index, value, offset = 0):
        raise NotImplementedError("not valid for this class")

    @staticmethod
    def key_walk(stack, key):
        """
        Set up stack to element with key.
        """
        node = stack[-1]
        while node.key != key:
            right = key > node.key
            node = node.leaf[right].node
            if node is None:
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
            # if cmp(stack[-2].key, key) * cmp(stack[-1].key, key) < 0:
            if stack[-2].key < key and stack[-1].key > key:
                break
            if stack[-2].key > key and stack[-1].key < key:
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

    def find_pos(self, key):
        if self.key == key:
            return self.leaf[0].count
        if key < self.key:
            return self.leaf[0].find_pos(key)
        return self.leaf[0].count + 1 + self.leaf[1].find_pos(key)
        # right = key > self.key
        # pos = (self.leaf[0].count + 1) if right else 0
        # return pos + self.leaf[right].find_pos(key)

    def twig(self):
        return "{:>2}[{:>2}]({}): {}".format(
            self.key,
            self.balance(),
            self.count(),
            self.value)

    def copy(self, leaf, deepcopy = False):
        """
        copy - shallow or deep
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
        leaf.count = leaf.node.count()
        leaf.depth = leaf.node.depth()

    def __repr__(self):
        return self.__class__.__name__ + "({!r}: {!r}; [{!r}, {!r}])".format(
            self.key, self.value, self.leaf[0], self.leaf[1])

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


class AVL_key_index_Node(AVL_keyed_Node):
    """
    AVL Tree node obejct with key.

    This contains the routines that require a key
    """
    __slots__ = ("value", "keys", "leaf", "chains")
    def __init__(self,
                 keys,
                 value = None,
                 chains = None):
        self.keys = list(keys)
        if chains is None:
            chains = len(self.keys)
        assert len(keys) == len(chains)
        self.chains = chains
        self.value = value
        self.new_leaf()

    def new_leaf(self):
        self.leaf = [AVL_Leaf() for i in range(2*self.chains)]

    def count(self, chain = 0):
        return 1 + self.leaf[chain].count + self.leaf[chain + 1].count

    def depth(self, chain = 0):
        return 1 + max(self.leaf[chain].depth, self.leaf[chain + 1].depth)

    def balance(self, chain = 0):
        return self.leaf[chain].depth - self.leaf[chain + 1].depth

    @staticmethod
    def first_walk(stack, right = False, chain = 0):
        """
        Set up stack to leftmost or rightmost node.
        """
        leaf = stack[-1].leaf[chain + right]
        while not leaf.empty():
            stack.append(leaf.node)
            leaf = leaf.node.leaf[chain + right]

    @classmethod
    def step_walk(cls, stack, right = False, chain = 0):
        """
        Make one step coming from [right] direction using the provided stack.
        """
        leaf = stack[-1].leaf[chain + (not right)]
        if not leaf.empty():
            stack.append(leaf.node)
            cls.first_walk(stack, right, chain)
        else:
            node = stack.pop()
            while (len(stack) > 0) and (stack[-1].leaf[chain + (not right)].node == node):
                node = stack.pop()


    @staticmethod
    def step_walk_n(stack, n, chain = 0):
        """
        Make n steps, positive for rightward, negative for leftward, the provided stack.
        """
        right = n > 0
        leaf = stack[-1].leaf[chain + right]
        while abs(n) > leaf.count:
            sig = bool2sign(right)
            n -= leaf.count * sig
            node = stack.pop()
            if len(stack) == 0:
                return
            leaf = stack[-1].leaf[chain + right]
            if node == leaf.node:
                n += leaf.count * sig
            else:
                n -= sig
            right = n > 0
            leaf = stack[-1].leaf[chain + right]
        while n != 0:
            stack.append(leaf.node)
            n -= (stack[-1].leaf[chain + (not right)].count + 1) * bool2sign(right)
            if n != 0:
                right = n > 0
                leaf = stack[-1].leaf[chain + right]
        # while n != 0:
        #     stack.append(stack[-1].leaf[chain + (n > 0)].node)
        #     n -= (stack[-1].leaf[chain + (n < 0)].count + 1)  * cmp(n, 0)


    @staticmethod
    def index_walk(stack, index, chain = 0):
        """
        Set up stack to element with index.
        """
        node = stack[-1]
        pos = node.leaf[chain].count
        while index != pos:
            right = index > pos
            node = node.leaf[chain + right].node
            stack.append(node)
            pos += bool2sign(right)*(node.leaf[chain + (not right)].count + 1)

    def pop(self, right, chain):
        if self.leaf[chain + right].empty():
            return self, self.leaf[chain + (not right)]
        else:
            node, self.leaf[chain + right] = self.leaf[chain + right].pop(right, chain)
            return node, self.rebalance(chain)

    def pop_deep(self, chain = 0):
        if self.count(chain) == 1:
            return self, AVL_Leaf()
        if self.balance(chain) == 0:
            right = self.leaf[chain].count <= self.leaf[chain + 1].count
        else:
            right = self.balance(chain) < 0
        node, self.leaf[chain + right] = self.leaf[chain + right].pop_deep(chain)
        return node, self.rebalance(chain)

    def insert_index(self, index, value, offset = 0, chain = 0):
        """
        Insert node by index.

        WARNING:
          This should only be used if no sorting is desired,
          that is, when implementing an *ordered* dictionary
          rather than a *sorted* dictionary.
        """
        assert chain == 0, "wrong chain"

        pos = offset + self.leaf[chain].count
        right = index > pos
        if right:
            offset = pos + 1
        if self.leaf[chain + right].empty():
            self.leaf[chain + right] = AVL_Leaf.new(value, key_index = True)
        else:
            self.leaf[chain + right].insert_index(index, value, offset, chain)
            self.leaf[chain + right] = self.leaf[chain + right].rebalance(chain)

    def delete_index(self, index, offset = 0, chain = 0):
        """
        Delete node by index.
        """
        pos = offset + self.leaf[chain].count
        if index == pos:
            if self.count(chain) == 1:
                return self, AVL_Leaf()
            if self.balance(chain) == 0:
                right = self.leaf[chain].count <= self.leaf[chain + 1].count
            else:
                right = self.balance(chain) < 0
            node, leaf = self.leaf[chain + right].pop(not right, chain)

            node.leaf[chain + (not right)] = self.leaf[chain + (not right)]
            node.leaf[chain + right] = leaf
            self.leaf[chain:chain+2] = [new_leaf(), new_leaf()]
            return self, node.rebalance(chain)
        right = index > pos
        if right:
            offset += self.leaf[chain].count + 1
        node, self.leaf[chain + right] = self.leaf[chain + right].node.delete_index(index, offset, chain)
        return node, self.rebalance(chain)

    def rotation(self, right, chain):
        """
        Rotate tree around self.
        """
        node = self.leaf[chain + (not right)].node
        self.leaf[chain + (not right)] = node.leaf[chain + right]
        node.leaf[chain + right] = AVL_Leaf(self)
        return AVL_Leaf(node)

    def rebalance(self, chain):
        if -1 <= self.balance(chain) <= +1:
            return AVL_Leaf(self)
        right = self.balance(chain) == -2
        if self.leaf[chain + right].balance(chain)*bool2sign(right) > 0:
            self.leaf[chain + right] = self.leaf[chain + right].rotation(right, chain)
        return self.rotation(not right, chain)

    def find_index(self, index, offset = 0, chain = 0):
        pos = offset + self.leaf[chain].count
        if index == pos:
            return self
        right = index > pos
        if right:
            offset += self.leaf[chain].count + 1
        return self.leaf[chain + right].node.find_index(index, offset, chain)

    def find_end(self, right = True, chain = 0):
        if self.leaf[chain + right].empty():
            return self
        return self.leaf[chain + right].find_end(right, chain)

    def find_value(self, value, chain):
        # not sure we really need that ...
        if self.value == value:
            return self
        for right in (False, True):
            if not self.leaf[chain + right].empty():
                node = self.leaf[chain + right].find_value(value, chain)
                if node is not None:
                    return node
        return None

    def get_nodes(self, nodes = None, chain = 0):
        if not self.leaf[chain].empty():
            self.leaf[chain].get_nodes(nodes, chain)
        nodes.append(self)
        if not self.leaf[chain + 1].empty():
            self.leaf[chain + 1].get_nodes(nodes, chain)

    def find_value_pos(self, value, chain):
        if self.value == value:
            return self.leaf[chain].count
        for right in (False, True):
            if not self.leaf[chain + right].empty():
                pos = self.leaf[chain + right].find_value_pos(value, chain)
                if pos >= 0:
                    if right:
                        return pos + self.leaf[chain].count + 1
                    return pos
        return -1

    def copy(self, leaf, deepcopy = False):
        """
        shallow copy
        """
        if deepcopy:
            leaf.node = self.__class__(
                deepcopy(self.keys),
                deepcopy(self.value))
        else:
            leaf.node = self.__class__(
                self.keys,
                self.value)
        for leaf_index in range(2*self.chains):
            if self.leaf[leaf_index].empty():
                leaf.node.leaf[leaf_index] = AVL_Leaf()
            else:
                self.leaf[leaf_index].copy(leaf.node.leaf[leaf_index])
        leaf.count = leaf.node.count()
        leaf.depth = leaf.node.depth()


    @staticmethod
    def key_walk(stack, key, chain = 0):
        """
        Set up stack to element with key.
        """
        assert chain < self.chains, "invalid chain"

        node = stack[-1]
        while node.key[chain] != key:
            right = key > node.key[chain]
            node = node.leaf[chain + right].node
            if node is None:
                break
            stack.append(node)

    # This seems to be efficient rarely
    @classmethod
    def step_walk_key(cls, stack, key, chain = 0):
        """
        Steps to key provided stack.
        """
        assert chain < self.chains, "invalid chain"

        while len(stack) > 1:
            if stack[-1].key == key:
                break
            # if cmp(stack[-2].key, key) * cmp(stack[-1].key, key) < 0:
            if stack[-2].key < key and stack[-1].key > key:
                break
            if stack[-2].key > key and stack[-1].key < key:
                break
            stack.pop()
        cls.key_walk(stack, key, chain)


    def find(self, key, chain = 0):
        """
        Find node with given key.
        """
        assert chain < self.chains, "invalid chain"

        if self.key[chain] == key:
            return self
        else:
            right = key > self.key[chain]
            if self.leaf[chain + right].empty():
                return None
            else:
                return self.leaf[chain + right].find(key, chain)

    def insert(self, key, value, chain = 0):
        """
        insert key in sorted order
        """
        assert chain < self.chains, "invalid chain"

        if key == self.key[chain]:
            self.value = value
        else:
            right = key > self.key[chain]
            if self.leaf[chain + right].empty():
                self.leaf[chain + right] = AVL_Leaf.new(key, value)
            else:
                self.leaf[chain + right].insert(key, value, chain)
                self.leaf[chain + right] = self.leaf[chain + right].rebalance(chain)

    def delete(self, key, chain = 0):
        """
        delete key
        """
        assert chain < self.chains, "invalid chain"

        if key == self.key[chain]:
            if self.count(chain) == 1:
                return self, AVL_Leaf()
            if self.balance(chain) == 0:
                right = self.leaf[chain].count <= self.leaf[chain + 1].count
            else:
                right = self.balance(chain) < 0
            node, leaf = self.leaf[chain + right].pop(not right)
            node.leaf[chain + right] = leaf
            node.leaf[chain + (not right)] = self.leaf[not right]
            self.leaf[chain:chain+2] = [new_leaf(), new_leaf()]
            return self, node.rebalance(chain)
        right = key > self.key[chain]
        if self.leaf[chain + right].empty():
            return None, AVL_Leaf(self)
        else:
            node, self.leaf[chain + right] = self.leaf[chain + right].delete(key, chain)
            return node, self.rebalance(chain)

    def find_pos(self, key, chain = 0):
        """
        find index of key
        """
        assert chain < self.chains, "invalid chain"

        if self.key[chain] == key:
            return self.leaf[chain].count
        if key < self.key[chain]:
            return self.leaf[chain].find_pos(key, chain)
        return self.leaf[chain].count + 1 + self.leaf[chain + 1].find_pos(key, chain)
        # right = key > self.key[chain]
        # pos = (self.leaf[chain].count + 1) if right else 0
        # return pos + self.leaf[chain + right].find_pos(key, chain)

    def twig(self, chain = 0, offset = 0):
        assert chain < self.chains, "invalid chain"
        return "{:>2}:{3i}[{:>2}]({}): {}".format(
            self.key[chain],
            self.leaf[chain].count(chain) + offset,
            self.balance(chain),
            self.count(chain),
            self.value)

    def tree(self, s, chain = 0, offset = 0):
        """get sting for tree"""
        assert chain < self.chains, "invalid chain"
        tree = ''
        index = 0
        s1 = self.twig(chain)
        s2 = "".ljust(4)
        if not self.leaf[chain + 1].empty():
            tree += self.leaf[chain + 1].tree(s + s2, chain, self.leaf[chain].offset(chain) + 1)
        tree += s + s1 + "\n"
        if not self.leaf[chain].empty():
            tree += self.leaf[chain].tree(s + s2, chain, offset)
        return tree


class List(collections.abc.MutableSequence):
    """
    Interface for AVL Tree list that allows O(log N) insert/remove.

    In contrast to ordinary lists where insert/deletation is O(N).

    Eventually, this should include general slicing for
    insertaion/deletation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize AVL tree (from iterable, if provided).
        """
        self.root = AVL_Leaf()
        self.append(*args)

    def append(self, *args):
        """
        Append elements at end of list.

        If only one argument is given, it will be interpreted as a
        list.
        """
        if len(args) == 1:
            if isinstance(args[0], collections.Iterable):
                for v in args[0]:
                    self.insert(v)
                return
            self.insert(args[0])
            return
        for v in args:
            self.insert(v)

    @classmethod
    def fromlist(cls, iterable):
        """
        Create new List object from iterable.
        """
        lst = cls()
        for value in iterable:
            lst.append(value)
        return lst

    def insert(self, *args):
        """
        Insert value before index, at end if index is not given.

        if argument list length is
        1 : append this item or list of items
            OK, this is problematic if we want to add a list.
            In this case an index needs to be provided
        2 : index and value, or list of indices and values
        3+: just append values

        This behaviour needs to be checked in parctice.
        """
        if len(args) == 1:
            if isinstance(args[0], collections.Iterable):
                for v in args[0]:
                    self._insert(self.root.count, v)
                return
            self._insert(self.root.count, args[0])
        elif len(args) == 2:
            if isinstance(args[0], collections.Iterable) and\
                   isinstance(args[1], collections.Iterable):
                for i,v in zip(args[0],args[1]):
                    self._insert(i, v)
                return
            self._insert(*args)
        else:
            for v in args:
                self._insert(self.root.count, v)

    def _insert(self, index, value):
        if self.root.empty():
            if index is not None and index != -1 and index != 0:
                raise IndexError("out of bounds")
            self.root = self.root.new(value)
            return
        # add range checks
        if index < 0:
            index = self.root.count - index
        self.root.insert_index(index, value)
        self.root = self.root.rebalance()

    # copy from dict type below
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

    # copy from dict type below
    def delete_index(self, index):
        if index < 0:
            index += self.root.count
        if not 0 <= index < self.root.count:
            raise IndexError("index out of range")
        node, self.root = self.root.delete_index(index)
        del node

    # copy from dict type below
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

    # copy from dict type below
    def iternodes_indexslice(self, *args):
        if self.root.empty():
            raise StopIteration("empty dictionary")
        if len(args) == 1 and issubclass(type(args[0]), slice):
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
            for i in range(start, stop, sign(step)):
                if (i - start) % step == 0:
                    yield stack[-1]
                self.root.node.step_walk(stack, reverse)
                if len(stack) == 0:
                    break

    # *modified* copy from dict type below
    def copy(self):
        root = AVL_Leaf()
        self.root.copy(root)
        duplicate = self.__class__()
        duplicate.root = root
        return duplicate

    def _get_slice(self, s):
        return self.__class__(node.value for node in self.iternodes_indexslice(s))

    def _set_slice(self, s, values):
        for node, value in zip(self.iternodes_indexslice(s), values):
            node.value = value

    def _del_slice(self, s, values):
        raise NotImplementedError()

    # abc/MutableSequence interface
    def __len__(self):
        return self.root.count

    def __size__(self):
        return self.root.depth

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            self._set_slice(index, value)
            return
        self.node_index(index).value = value

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._get_slice(index)
        return self.node_index(index).value

    def __delitem__(self, index):
        if isinstance(index, slice):
            self._del_slice(index)
            return
        return self.delete_index(index)

    def __contains__(self, value):
        for node in self.root.iternodes():
            if node.value == value:
                return True
        return False

    def __iter__(self):
        for node in self.root.iternodes():
            yield node.value

    def __repr__(self):
        return (self.__class__.__name__ +
                "([" + ", ".join(repr(node.value) for node in self.root.iternodes())+"])")

    def __add__(self, other):
        assert isinstance(other, collections.Iterable)
        return self.copy().insert(other)

    def index(self, value, *args):
        assert len(args) <= 2
        limits = [0, self.root.count]
        limits[0:len(args)] = args
        s = slice(*limits)
        pos = limits[0]
        for node in self.iternodes_indexslice(s):
            if node.value == value:
                return pos
            pos += 1
        raise ValueError("{!r} is not in list".format(value))

    def __mul__(self, n):
        assert isinstance(n, int)
        assert n >= 0
        new = self.__class__()
        for i in range(n):
            new.insert(self)
        return new

    __str__ = __repr__
    __rmul__ = __mul__

class SortedDict(collections.abc.MutableMapping):
    """
    Interface for AVL Tree sorted dictionary that also is indexable.

    In contrast to ordinary dictionary, the keys need to be comparable
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
        self.type = kwargs.pop('type', None)
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
        """
        Create dictionary from keys provided as iterable dictionary type
        """
        dictionary = cls()
        for key in iterable:
            dictionary[key] = value
        return dictionary

    def setdefault(self, key, value = None):
        if self.root.empty():
            self.root = self.root.new(key, value)
        else:
            item = self.root.get_key(key)
            if item is None:
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
        if node is None:
            return value
        else:
            return node.value

    def pop(self, key, value = None):
        if self.root.empty():
            return value
        item = self.delete(key)
        if item is None:
            return value
        return item

    def popitem(self):
        if self.root.empty():
            return value
        node, self.root = self.root.pop_deep()
        if node is None:
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
        duplicate.type = self.type
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
        # this may need to be generalized to return node or key as well
        if self.root.empty():
            node = None
        else:
            node, self.root = self.root.delete(key)
        if node is None:
            return None, None
        else:
            return node.key, node.value

    def delete_index(self, index):
        # this may need to be generalized to return node or key as well
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

    def iternodes_step(self, step, right = False):
        for node in self.root.iternodes_n(step, (-self.root.count + 1)*int(right)):
            yield node
    def iternodes_slice(self, slc):
        for node in self.root.iternodes_slice(slice(*slc.indices(self.root.count)),0):
            yield node


    def itervalues_indexlist(self, *args, **kwargs):
        """
        Return iterator of values for given list of indices.
        """
        for node in self.iternodes_indexlist(*args, **kwargs):
            yield node.value

    def itervalues_keylist(self, *args, **kwargs):
        """
        Return iterator of values for given list of keys
        """
        for node in self.iternodes_keylist(*args, **kwargs):
            yield node.value

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

            for i in range(start, stop, sign(step)):
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
        elif step is None:
            step = 1
        reverse = step < 1
        stack = [self.root.node]
        if start is None:
            self.root.node.first_walk(stack, reverse)
        else:
            self.root.node.key_walk(stack, start)
            if (stack[-1].key != start) and \
                    operator.xor(stack[-1].key < start, reverse):
                self.root.node.step_walk(stack, reverse)
                if len(stack) == 0:
                    raise StopIteration()
        if abs(step) == 1:
            while operator.xor(stack[-1].key < stop, reverse) or (stop is None):
                yield stack[-1]
                self.root.node.step_walk(stack, reverse)
                if len(stack) == 0:
                    break

        # threshold to adjust
        elif abs(step) <= 2:
            i = 0
            while operator.xor(stack[-1].key < stop, reverse) or (stop is None):
                if i % step == 0:
                    yield stack[-1]
                self.root.node.step_walk(stack, reverse)
                if len(stack) == 0:
                    break
                i += 1
        else:
            while operator.xor(stack[-1].key < stop, reverse) or (stop is None):
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

    # replaced for python3 versions that retunr "views"
    # # INEFFICIENT
    # def nodes(self):
    #     nodes = []
    #     if not self.root.empty():
    #         self.root.get_nodes(nodes)
    #     return nodes

    # # INEFFICIENT
    # def items(self):
    #     return [(node.key, node.value) for node in self.nodes()]

    # # INEFFICIENT
    # def keys(self):
    #     return [node.key for node in self.nodes()]

    # # INEFFICIENT
    # def values(self):
    #     return [node.value for node in self.nodes()]

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
        if node is None:
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
        if node is None:
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


    def has_value(self, value):
        if self.root.empty():
            return False
        return self.root.find_value(value) is not None

    # explicity index by index not key
    def __call__(self, index):
        return self.node_index(index).value

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
        if (self.type is None) and (isint == False):
            iskey == True
        if iskey == True:
            return self.values_keyslice(s)
        return self.values_indexslice(s)

    def _set_indexslice(self, s, values):
        for node, value in zip(self.iternodes_indexslice(s), values):
            node.value = value

    def _set_keyslice(self, s, values):
        for node, value in zip(self.iternodes_keyslice(s), values):
            node.value = value

    def _set_slice(self, s, value):
        assert issubclass(type(s), slice), "need slice type"
        args = (s.start, s.stop, s.step)
        iskey = False
        isint = False
        for arg in args:
            if issubclass(type(arg), int):
                isint = True
            if self.type is not None:
                if issubclass(type(arg), self.type):
                    iskey = True
        if (self.type is None) and (isint == False):
            iskey == True
        if iskey == True:
            self._set_keyslice(s, value)
            return
        self._set_indexslice(s, value)

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
        if (self.type is None) and (isint == False):
            iskey == True
        if iskey == True:
            self._del_keyslice(s)
            return
        self._del_indexslice(s)


    # list interface

    def pop_end(self, right = True):
        if self.root.empty():
            return value
        node, self.root = self.root.pop(right)
        if node is None:
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
        duplicate.root = self.type
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

    # dictionary default/abc interface

    __contains__ = has_key

    def __len__(self):
        return self.root.count

    def __size__(self):
        return self.root.depth

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._get_slice(key)
        if isinstance(key, collections.Iterable):
            if self.type is not None:
                if not isinstance(key, self.type):
                    if len(key) != 1:
                        return [self.__getitem__(ikey) for ikey in key]
                    else:
                        key = key[0]
        try:
            return self.node_key(key).value
        except:
            return self.node_index(key).value
        # if self.type is not None:
        #     if isinstance(key, self.type):
        #         return self.node_key(key).value
        # return self.node_index(key).value

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self._set_slice(key, value)
            return
        if issubclass(type(key), collections.Iterable):
            if self.type is not None:
                if not isinstance(key, self.type):
                    if len(key) != 1:
                        for xkey, xvalue in zip(key, value):
                            self.__setitem__(xkey, xvalue)
                        return
                    else:
                        key = key[0]
        try:
            self.insert(key, value)
        except:
            self.node_index(key).value = value
        # if self.type is not None:
        #     if isinstance(key, self.type):
        #         self.insert(key, value)
        #         return
        # self.node_index(key).value = value

    def __delitem__(self, key):
        if isinstance(key, slice):
            self._del_slice(key)
            return
        if issubclass(type(key), collections.Iterable):
            if self.type is not None:
                if not isinstance(key, self.type):
                    if len(key) != 1:
                        for xkey in key:
                            self.__delitem__(xkey)
                        return
                    else:
                        key = key[0]
        try:
            self.delete(key)
        except:
            self.delete_index(key)
        # if self.type is not None:
        #     if issubclass(key, self.type):
        #         self.delete(key)
        #         return
        # self.delete_index(key)

    def __str__(self):
        return self._name()

    def __repr__(self):
        return self.__class__.__name__ + "("+self._name()+")"

    def __iter__(self):
        return self.iterkeys()

    def nodes(self):
        return SortedDictNodesView(self)

    def items(self):
        return SortedDictItemsView(self)

    def keys(self):
        return SortedDictKeysView(self)

    def values(self):
        return SortedDictValuesView(self)


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

class SortedDictNodesView(collections.abc.KeysView):
    def __init__(self, parent):
        self._mapping = parent
    def __len__(self):
        return len(self._mapping)
    def __iter__(self):
        return self._mapping.iternodes()
    def __contains__(self, key):
        return key in self._mapping
    def __repr__(self):
        return self.__class__.__name__ + "([" + ", ".join(
            repr(n) for n in self._mapping.iternodes()) + "])"

class SortedDictItemsView(collections.abc.ItemsView):
    def __init__(self, parent):
        self._mapping = parent
    def __len__(self):
        return len(self._mapping)
    def __iter__(self):
        return self._mapping.iteritems()
    def __contains__(self, key):
        return key in self._mapping
    def __repr__(self):
        return self.__class__.__name__ + "([" + ", ".join(
            "({!r}, {!r})".format(k, v) for k,v in self._mapping.iteritems()) + "])"

class SortedDictKeysView(collections.abc.KeysView):
    def __init__(self, parent):
        self._mapping = parent
    def __len__(self):
        return len(self._mapping)
    def __iter__(self):
        return self._mapping.iterkeys()
    def __contains__(self, key):
        return key in self._mapping
    def __repr__(self):
        return self.__class__.__name__ + "([" + ", ".join(
            repr(k) for k in self._mapping.iterkeys()) + "])"

class SortedDictValuesView(collections.abc.ValuesView):
    def __init__(self, parent):
        self._mapping = parent
    def __len__(self):
        return len(self._mapping)
    def __iter__(self):
        return self._mapping.itervalues()
    def __contains__(self, value):
        return self._mapping.has_value(value)
    def __repr__(self):
        return self.__class__.__name__ + "([" + ", ".join(
            repr(v) for v in self._mapping.itervalues()) + "])"

def test():
    import random
    a = SortedDict()
    random.seed(0)
#    for i in random.sample(range(0,999),500):
#    for i in range(7):
    for i in range(25,-1,-1):
#    for i in range(0,26):
#        print("Adding: {}".format(i))
        a.insert(str(chr(64+2*i+1)),"{:06X}".format(i))
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

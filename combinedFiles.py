import numpy as np

class Node(object):
    def __init__(self, key, val=None):
        self.key = key
        self.value = val
        self.left = None
        self.right = None
        self.height = 0


class AVLTree(object):
    def __init__(self, node=None):
        self.root = node

    def height(self, node):
        if node is None:
            return -1
        return node.height

    def rotate(self, node):
        left_height = self.height(node.left)
        right_height = self.height(node.right)
        if left_height > right_height:
            ll_height = self.height(node.left.left)
            lr_height = self.height(node.left.right)
            if ll_height > lr_height:
                return self.rotateRight(node)
            else:
                node.left = self.rotateLeft(node.left)
                return self.rotateRight(node)
        else:
            rl_height = self.height(node.right.left)
            rr_height = self.height(node.right.right)
            if rr_height > rl_height:
                return self.rotateLeft(node)
            else:
                node.right = self.rotateRight(node.right)
                return self.rotateLeft(node)

    def rotateLeft(self, node):
        right_node = node.right
        node.right = right_node.left
        right_node.left = node
        node.height = max(self.height(node.left), self.height(node.right)) + 1
        right_node.height = max(self.height(right_node.left), self.height(right_node.right)) + 1
        return right_node

    def rotateRight(self, node):
        left_node= node.left
        node.left = left_node.right
        left_node.right = node
        node.height = max(self.height(node.left), self.height(node.right)) + 1
        left_node.height = max(self.height(left_node.left), self.height(left_node.right)) + 1
        return left_node


    def insertAtNode(self, node, key, value):
        if node is None:
            node = Node(key, value)
            return node, node, True

        if node.key == key:
            node.value = value
            return node, node, False
        elif node.key > key:
            node.left, nd, inserted = self.insertAtNode(node.left, key, value)
            node.height = max(self.height(node.left), self.height(node.right)) + 1
            if abs(self.height(node.left) - self.height(node.right)) > 1:
                node = self.rotate(node)
            return node, nd, inserted
        else:
            node.right, nd, inserted = self.insertAtNode(node.right, key, value)
            node.height = max(self.height(node.left), self.height(node.right)) + 1
            if abs(self.height(node.left) - self.height(node.right)) > 1:
                node = self.rotate(node)
            return node, nd, inserted


    def insert(self, key, value=None):
        vals = self.insertAtNode(self.root, key, value)
        self.root = vals[0]
        return vals[1:] # actual node inserted, bool
import numpy as np

class Node(object):
    def __init__(self, attr, threshold, left_child=None, right_child=None):
        self.attr = attr
        self.threshold = threshold
        self.left = left_child
        self.right = right_child


class GBoostClassificDecTree(object):
    def __init__(self, max_leaves):
        self.maxLeaves = max_leaves
        self.roots = []

    def buildTree(self, inputs, outputs, features):
        if len(features) == 0:
            return None
        feature = features[-1]
        rem_feat = features[0:-1]
        
        vals = sorted(list(set(inputs[:,feature])))
        max_red = None
        best_threshold = None
        for threshold in vals:
            gini_imp_red = self.getGiniImpuriyReduction(threshold, inputs[:, feature], y_out)
            if (max_red is None) or (max_red < gini_imp_red):
                max_red = gini_imp_red
                best_threshold = threshold
            # build the tree
        node = Node(feature, best_theshold)
        left_data = np.where(inputs[:, feature] <= best_threshold, True, False)
        node.left, rem_feat = self.buildTree(inputs[left_data,:], outputs[left_data], rem_feat)
        right_data = np.where(left_data, False, True)
        node.right, rem_feat = self.buildTree(inputs[right_data, :], outputs[right_data], rem_feat)

        return node, rem_feat


    def constructTree(self, inputs, outputs):
        ninp = inputs.shape[1]
        max_levels = np.log2(ninp)
        features = np.random.choice(ninp, size=max_levels, replace=False)
        # assume categorical outputs, continuous inputs
        # classification
        output_types = list(set(outputs))
        if len(output_types) > 2:
            for output in output_types:
                y_out = np.where(outputs == output, True, False)
                self.roots.append(self.buildTree(inputs, y_out, features))
        else:
            y_out = np.where(outputs == output_types[0], True, False)
            self.roots.append(self.buildTree(inputs, y_out, features))


import numpy as np
from collections import deque

class Edge(object):
    def __init__(self, v, capacity, flow=0):
        self.v = v
        self.flow = flow
        self.capacity = capacity
        assert self.flow <= self.capacity

    def __hash__(self):
        return hash(self.v)

    def __eq__(self, other):
        return self.v == other.v


# graph is a dict of dict {u: {v: Edge(u->v)}}

# O(E^2V)
class EdmundKarp(object):
    def BFS(self, graph, s, t):
        queue = deque()
        queue.append(s)
        nvtx = len(graph)
        parent = np.zeros(nvtx, dtype=np.int)
        parent[:] = -1
        parent[s] = -2
        while len(queue):
            vertex = queue.popleft()
            for edge in graph.get(vertex, {}):
                if (parent[edge.v] == -1) and (edge.capacity > edge.flow):
                    parent[edge.v] = vertex
                    queue.append(edge.v)
                    if edge.v == t:
                        break
        return parent

    def augmentflow(self, graph, s, t, parent):
        v = t
        min_flow = -1
        while v != s:
            u = parent[v]
            edge = graph[u][v]
            if (min_flow < 0) or (min_flow > edge.capacity - edge.flow):
                min_flow = edge.capacity - edge.flow
            v = u

        v = t
        while v != s:
            u = parent[v]
            edge = graph[u][v]
            edge.flow += min_flow
            # reverse edge
            if v not in graph:
                graph[v] = dict()
            if u not in graph[v]:
                graph[v][u] = Edge(u, edge.flow, 0)
            graph[v][u].capacity = edge.flow
            v = u
        return min_flow


    def maxflow(self, graph, s, t):
        parent = self.BFS(graph, s, t)
        flow = 0
        while parent[t] != -1:
            flow += self.augmentflow(graph, s, t, parent)
        return flow
import numpy as np

class Node(object):
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.marked = False
        self.children = []
        self.nextSibling = None
        self.prevSibling = None
        self.parent = None


class FibonacciHeap(object):
    def __init__(self):
        self.root = None
        self.keyDict = {}


    def add(self, key, value):
        if key in self.keyDict:
            return False

        if self.root is None:
            self.root = Node(key, value)
            self.root.nextSibling = self.root
            self.root.prevSibling = self
            self.keyDict[key] = self.root
            return True

        node = Node(key, value)
        nextsib = self.root.nextSibling
        self.root.nextSibling = node
        node.nextSibling = nextsib
        nextsib.prevSibling = node
        node.prevSibling = self.root
        if self.root.value > value:
            self.root = node

        self.keyDict[key] = node
        return True

    def rebalanceTrees(self):
        nodes = []
        node = self.root
        nodes.insert(len(node.children), node)
        node = node.nextSibling
        while node != self.root:
            index = len(node.children)
            smaller_node = node
            while (len(nodes) > index)) and (isinstance(nodes[index], Node)):
                other_node = nodes[index]
                if smaller_node.value > other_node.value:
                    smaller_node = other_node
                    other_node = node
                smaller_node.children.append(other_node)
                other_node.prevSibling = None
                other_node.nextSibling = None
                other_node.parent = smaller_node
                nodes[index] = None
                index = len(smaller_node.children)
                
            nodes.insert(index, smaller_node)

        min_node = None
        prev_node = None
        first_node = None
        for i in range(len(nodes)):
            if nodes[i]:
                if min_node is None:
                    min_node = nodes[i]
                elif nodes[i].value < min_node.value:
                    min_node = nodes[i]

                if prev_node:
                    prev_node.nextSibling = nodes[i]
                    nodes[i].prevSibling = prev_node
                prev_node = nodes[i]
                if first_node is None:
                    first_node = nodes[i]

        self.root = min_node
        first_node.prevSibling = prev_node
        prev_node.nextSibling = first_node


    def meldInRootList(self, node, update_root=True):
        node.parent = None
        nextsib = self.root.nextSibling
        self.root.nextSibling = node
        node.prevSibling = self.root
        node.nextSibling = nextsib
        nextsib.prevSibling = node

        if update_root:
            if self.root.value > node.value:
                self.root = node

    def pop(self):
        if self.root is None:
            raise ValueError("Empty heap")
        
        poped_node = self.root
        for node in self.root.children:
            self.meldInRootList(node, update_root=False)
        
        self.rebalanceTrees()
        return popped_node


    def update(self, key, new_value):
        if key not in self.keyDict:
            raise ValueError("Key %s not found in heap"%str(key))

        node = self.keyDict[key]
        node.value = new_value

        if node.parent is not None:
            if node.parent.value > new_value:
                if not node.parent.marked:
                    node.parent.marked = True
                    self.meldInRootList(node)
                else:
                    node.parent.marked = False
                    self.meldInRootList(node)
                    self.meldInRootList(node.parent)
        else:
            if node.value < self.root.value:
                self.root = node



import numpy as np

# fod fulkerson algorithm
# graph is adjacency matrix-list using a list of dicts

class Edge(object):
    def __init__(self, v, capacity, flow=0):
        self.v = v
        self.capacity = capacity
        self.flow = flow
        assert self.flow <= self.capacity


class FordFulkerson(object):
    def DFS(self, graph, source, sink, parent):
        parent[:] = -1
        parent[source] = -2
        stack = [source]
        while len(stack):
            u = stack.pop()
            for v in graph[u]:
                edge = graph[u][v]
                if (parent[v] == -1) and (edge.capacity > edge.flow):
                    parent[v] = u
                    stack.append(v)
                    if v == sink:
                        return

    def augmentflow(self, graph, source, sink, parent):
        flow = None
        v = sink
        if parent[sink] < 0:
            return 0
        while v != source:
            u = parent[v]
            edge = graph[u][v]
            if (flow is None) or (flow > edge.capacity - edge.flow):
                flow = edge.capacity - edge.flow

            v = u

        if flow is None:
            return 0

        v = sink
        while v != source:
            u = parent[v]
            edge = graph[u][v]
            edge.flow += flow
            # reverse edge
            if u not in graph[v]:
                graph[v] = {u: Edge(u, 0, 0)}
            graph[v][u].capacity = edge.flow
        return flow

    def maxflow(self, graph, source, sink):
        self.nVertex = len(graph)
        parent = np.ndarray(self.nVertex, dtype=int)
        self.DFS(graph, source, sink, parent)
        flow = 0
        while parent[sink] != -1:
            flow += self.augmentflow(graph, source, sink, parent)
            self.DFS(graph, source, sink, parent)

        return flow
import numpy as np
from collections import deque

class HopcroftKarp(object):
    def findLevelGraph(self, uConn, vConn, u_level, u_next, v_next):
        u_level[:] = -1
        level = -1
        queue = deque()
        for i in range(len(uConn)):
            if u_next[i] == -1:
                queue.append(i)
                u_level[i] = 0 
            

        while len(queue):
            u = queue.popleft()
            for v in uConn[u]:
                if v_next[v] != -1:
                    un = v_next[v]
                    if u_level[un] == -1:
                        queue.append(un)
                        u_level[un] = u_level[u] + 1
                else:
                    level = u_level[u] + 1
        return level


    def dfs(self, u, uConn, vConn, u_level, u_next, v_next, level_val):
        if u_next[u] != -1:
            return False

        if u_level[u] < level_val:
            return False

        for v in uConn[u]:
            if v_next[v] == -1:
                u_next[u] = v
                v_next[v] = u
                return True
            else:
                un = v_next[v]
                if u_level[un] == u_level[u] + 1:
                    found = dfs(un, uConn, vConn, u_level, u_next, v_next, u_level[un])
                    if found:
                        u_next[u] = v
                        v_next[v] = u
                        return True

        return False

    def maxflow(self, uConn, vConn):
        u_level = np.zeros(len(uConn), dtype=np.int)
        u_next = np.zeros(len(uConn), dtype=np.int)
        v_next = np.zeros(len(vConn), dtype=np.int)
        u_next[:] = -1
        v_next[:] = -1

        flow =0
        while self.findLevelGraph(uConn, vConn, u_level, u_next, v_next):
            for u in range(len(uConn)):
                if self.DFS(u, uConn, vConn, u_level, v_next, v_next, 0):
                    flow += 1

        return flow
import numpy as np

class KMP(object):
    def findLongestPrefix(self):
        for i in range(1, len(self.pattern)):
            j = i-1
            while j >= 0:
                last_len = self.lp[j]
                if self.pattern[i] == self.pattern[last_len+1]:
                    self.lp[i] = self.lp[last_len] + 1
                    break
                elif j > 0:
                    j = self.lp[j]
                else:
                    j = 0  # nt needed as j is 0 here
                    self.lp[i] = 0
                    break

    def find(self, word):
        res = []
        if len(word) < len(pattern):
            return res
        j = 0
        i = 0
        while i < len(word):
            while j >= 0:
                if j == len(self.pattern):
                    res.append(i)
                    j = self.lp[j]
                    break
                elif word[i] == self.pattern[j]:
                    j += 1
                    i += 1
                elif j != 0:
                    j = self.lp[j-1] + 1
                else:
                    i += 1
                    break

        return res

    def __init__(self, pattern):
        self.pattern = pattern
        self.lp = np.zeros(len(pattern), dtype=np.int)
        self.findLongestPrefix()
# implement a least recently used eviction cache

class LinkedListNode(object):
    def __init__(self, key, val, prev=None, nxt=None):
        self.key = key
        self.val = val
        self.prev = prev
        self.nxt = nxt

class LRUCache(object):
    def __init__(self, capacity):
        self.cache = {}
        self.begin = LinkedListNode(None, None)
        self.end = LinkedListNode(None, None)
        self.begin.next = self.end
        self.end.prev = self.begin
        self.cap = capacity

    def evict(self):
        last_nd = self.end.prev
        ll_nd = last_nd.prev
        self.end.prev = ll_nd
        ll_nd.next = self.end

        self.cache.pop(last_nd.key)  # del self.cache[last_nd.key]

    def moveToFront(self, node):
        old_next = self.begin.next
        nd.prev = self.begin
        nd.next = old_next
        old_next.prev = nd

    def insert(self, key, val):
        if key in self.cache:
            self.cache[key].val = val
            self.moveToFront(self.cache[key])
            return

        if len(self.cache) == self.capacity:
            self.evict()

        nd = LinkedListNode(key, val)
        self.movetoFront(nd)
        self.cache[key] = nd
        

    def get(self, key, default):
        if key not in self.cache:
            return default

        self.moveToFront(self.cache[key])
        return self.cache[key].val

import numpy as np

class NQueens(object):
    def validateSolution(self, last_row):
        for row in range(last_row):
            for r2 in range(row+1, last_row):
                if self.queenPos[row] == self.queenPos[r2]:
                    return False
                if abs(self.queenPos[r2] - self.queenPos[row]) == r2 - row:
                    return False

        return True

    def solutionFromRow(self, row):
        if row >= self.n:
            return self.validateSolution(self.n)

        for col in range(self.n):
            self.queenPos[row] = col
            if self.validateSolution(row+1):
                if self.solutionFromRow(row+1):
                    return True
        return False

    def nextSolution(self):
        for row in range(self.n):
            for col in range(self.n):
                self.queenPos[row] = col
                if self.validateSolution(row+1):
                    if self.solutionFromRow(row+1):
                        return True

        return False

    def allSoln(self, n):
        self.n = n
        self.queenPos = np.zeros(n, dtype=int)
        solutions = []
        while self.nextSolution():
            solutions.append(self.queenPos.copy())
        return solutions
# Return the number of unique solutions to N queens

class NQueens(object):
    def self.recCount(self, board, N, row, usedCols, diag1, diag2):
        if row == N:
            return 1

        count = 0
        for j in range(N):
            if (j in usedCols) or ((row-j) in diag1) or (row+j) in diag2):
                continue
            board[row] = j
            usedCols.add(j)
            diag1.add(row-j)
            diag2.add(row+j)
            count += self.recCount(board, N, row+1, usedCols, diag1, diag2)
            usedCols.remove(j)
            diag1.remove(row-j)
            diag2.remove(row+j)
        return count


    def numSoln(self, N):
        usedCols = set()
        diag1 = {}
        diag2 = {}

        board = np.zeros(N, dtype=np.int32)
        return self.recCount(board, N, 0, usedCols, diag1, diag2)
# count the number of positions in nqueens

class NQueens3(object):
    def numPositions(self, blen):
        self.boardLen = blen
        self.colSet = {}
        self.diag1 = {}
        self.diag2 = {}
        self.solnCount = 0
        self.recursiveCount(0)
        return self.solnCount

    def recursiveCount(self, row):
        if row == self.boardLen:
            self.solnCount += 1
            return
        for i in range(self.boardLen):
            if i in self.colSet or (row+i) in self.diag1 or (row-i) in self.diag2:
                continue
            self.colSet.add(i)
            self.diag1.add(row+i)
            self.diag2.add(row-i)
            self.recursiveCount(row+1)
            self.colSet.remove(i)
            self.diag1.remove((row+i))
            self.diag2.remove(row-i)
            
import numpy as np

class Node(object):
    def __init__(self, feature, threshold, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class RandomForest(object):
    ''' Construct a random forest for binary classification problem '''
    def __init__(self, ntrees, nsplits=None):
        ''' Initialize.
        :ntrees number of decision trees
        :nsplits number of features used to split tree nodes. Tree will have atmost nsplits height
        '''
        self.trees = [None]*ntrees
        self.nSplits = nsplits
        self.oobSamples = [None]*ntrees

    def _giniNode(self, outputs):
        pos = output.sum()
        neg = output.shape[0] - pos
        prob_pos = pos / float(output.shape[0])
        prob_neg = 1 - prob_pos
        return prob_pos*(1 - prob_pos) + prob_neg*(1 - prob_neg)


    def _getGiniImpRed(self, inputfeature, outputs, threshold):
        ''' GINI impurity reduction by splitting on feature at threshold '''
        gini = self._giniNode(outputs, threshold)
        left = (inputfeature < threshold)
        right = (inputfeature >= threshold)
        gini_left = self._giniNode(outputs[left])
        gini_right = self._giniNode(outputs[right])
        nobs = outputs.shape[0]
        return gini - left.sum()/float(nobs) * gini_left - right.sum()/float(nobs) * gini_right


    def _constructTree(self, inputs, outputs, features, splits):
        ''' Construct a decision tree
        :inputs 2 dimensional numpy ndarray of shape (num observations, num features)
        :outputs 1 dimensional ndarray with output. Shape (num_observations)
        :features list of features to split the node
        :splits threshold on number of splits
        ''' 
        if splits <= 0:
            return

        sel_feat = None
        reduction = None
        sel_threshold = None
        for feat in features:
            threshold = np.random.choice(inputs[:, feat], size=1)
            gini_red = self._getGiniImpRed(inputs[:, feat], outputs, threshold)
            if (reduction is None) or (reduction < gini_red):
                gini_red = reduction
                sel_feat = feat
                sel_threshold = threshold

        node = Node(sel_feat, sel_threshold)
        left_data = (inputs[:, sel_feat] <= sel_threshold)
        features_rem = [f for f in features if f != sel_feat]
        node.left = self._constructTree(inputs[left_data,:], outputs[left_data], features_rem, splits-1)
        right_data = np.logical_not(left_data)
        node.right = self._constructTree(inputs[right_data,:], outputs[right_data], features_rem, splits-1)
        return node


    def construct(self, inputs, outputs):
        # assume output is binary
        nfeat = inputs.shape[0]
        if self.nSplits is None:
            self.nSplits = int(np.sqrt(nfeat))
        y_labels = sorted(list(set(outputs)))
        assert len(y_labels) == 2
        y_out = np.where(outputs == y_labels[0], True, False)
        features = np.arange(inputs.shape[1])
        for i in len(self.trees):
            sample_inputs = np.random.choice(inputs.shape[0], inputs.shape[0], replace=True)
            
            self.trees[i] = self.constructTree(inputs[sample_inputs, :], y_out[sample_inputs], features)
            self.oobSamples[i] = sample_inputs


import copy
import numpy as np

'''
def nqueen(row):
    if row >= N:
        return validate(...)
    for col in range(N):
        self.queenPos[row] = col
        precheck
        self.nqueen(row+1)

'''

class Soduku(object):
    def validateBoard(self, row, col):
        used = np.zeros(9, dtype=bool)
        for i in range(row+1):
            if isinstance(self.board[i][col], str):
                val = self.board[i][col] - '0'
            else:
                val = self.board[i][col]
            if used[val]:
                return False
            used[val] = True

        used[:] = False
        for j in range(col+1):
            val = self.board[row][j]
            if isinstance(val, str):
                val = val - '0'
            if used[val]:
                return False
            used[val] = True

        used[:] = False
        if (row%3 == 0) and (col%3 == 0) and (row > 0) and (col > 0):
            for i in range(row-2, row+1):
                for j in range(col-2, col+1):
                    val = self.board[i][j]
                    if isinstance(val, str):
                        val = val - '0'
                    if used[val]:
                        return False
                    used[val] = True
        return True

    def findAllSolnsFrom(self, row, col):
        if (row, col) == self.board.shape:
            if self.validateBoard(row, col):
                self.solutions.append(copy.deepcopy(self.board))
            else:
                return

        if self.board[row][col] != '.':
            if col < len(self.board)-1:
                return self.findAllSolnsFrom(row, col+1)
            else:
                return self.findAllSolnsFrom(row+1, 0)
        else:
            for num in range(1, 10):
                self.board[row][col] = num
                if self.validateBoard(row, col):
                    if col == self.board.shape[1]-1:
                        self.findAllSolnsFrom(row+1, 0)
                    else:
                        self.findAllSolnsFrom(row, col+1)

    def findall(self, board):
        if len(board) == 0:
            return []

        self.solutions = []
        self.findAllSolutionsFrom(0,0)
        return self.solutions[]

import numpy as np

class TSP(object):

    def getSubtour(self, mask, city, tsp, ncities, dist_arr):
        if tsp[mask, city] >= 0:
            return tsp[mask, city]

        min_val = -1.0
        for i in range(1, ncities):
            if i == city:
                continue
            if mask & (1 << i):
                mask = mask - (1 << i)
                val = self.getSubtour(mask, i) + dist_arr[i, city]
                if (min_val < 0) or (min_val > val):
                    min_val = val
                mask = mask + (1 << i)
        tsp[mask, city] = min_val
        return min_val


    def shortestTour(self, dist_arr):
        # TSP(n, i): shortest tour starting at 0, ending at i, using cities in n mask
        #   = min(TSP(n-1, k) + dist[k,i] for k in n cities)
        ncities = dist_arr.shape[0]
        mask_sz = 2**ncities
        tsp = np.zeros((mask_sz, ncities), dtype=np.float)
        tsp[:,:] = -1
        tsp[0,0] = 0
        for i in range(1, ncities):
            tsp[0, i] = dist_arr[0, i]

        min_val = -1
        mask = mask_sz - 2

        for i in range(1, ncities):
            mask = mask - (1 << i)
            subtour_dist = self.getSubtour(mask, i, TSP)
            if (min_val < 0) or (min_val > subtour_dist + dist_arr[i,0]):
                min_val = subtour_dist + dist_arr[i,0]
            mask = mask + (1 << i)

        return min_val
import numpy as np

class Graph(object):
    def __init__(self, nvert):
        self.graph = [[]] * nvert  #adj list

    def addEdge(self, from_vt, to_vt, weight):
        self.graph[from_vt].append((to_vt, weight))

class SSSPDag(object):
    def __init__(self, graph):
        self.graph = graph

    def dfs(self, vtx, ordered_vert, indx, visited):
        if visited[vtx]:
            return indx
        visited[vtx] = True
        for nbr in self.graph.graph[vtx]:
            indx = self.dfs(nbr[0], ordered_vert, indx)
        ordered_vert[indx] = vtx
        return indx - 1

    def topSort(self, ordered_vert):
        nvert = len(self.graph.graph)
        visited = np.zeros(nvert, dtype=bool)
        indx = nvert-1
        for i in range(nvert):
            indx = self.dfs(i, ordered_vert, indx, visited)

    def findSSSP(self, source):
        nvert = len(self.graph.graph)
        result_dist = np.full(nvert, np.inf, dtype=np.int)
        result_dist[source] = 0
        ordered_vert = np.zeros(nvert, dtype=np.int)
        self.topSort(ordered_vert)

        indx = -1
        for i in range(nvert):
            if ordered_vert[i] == source:
                indx = i
                break

        for i in range(indx, nvert):
            vtx = ordered_vert[i]
            for nbr in self.graph.graph[vtx]:
                if result_dist[nbr[0]] > result_dist[vtx] + nbr[1]:
                    result_dist[nbr[0]] = result_dist[vtx] + nbr[1]

        return result_dist



class Node(object):
    def __init__(self, val, left_node=None, right_node=None):
        self.value = val
        self.leftNode = left_node
        self.rightNode = right_node

class RestoreOrder(object):
    def inorderTraverse(self, node, last_node, err_node1):
        if node.leftNode:
            last_node2, err_node1, err_node2 = self.inorderTraverse(node.leftNode, last_node, err_node1)
            if err_node2:
                return last_node2, err_node1, err_node2
            last_node = last_node2

        if last_node:
            if last_node.value > node.value:
                if err_node1:
                    return last_node, err_node1, node
                else:
                    err_node1 = last_node

        last_node = node
        if node.rightNode:
            last_node2, err_node1, err_node2 = self.inorderTraverse(node.rightNode, last_node, err_node1)
            if err_node2:
                return last_node2, err_node1, err_node2

            last_node = last_node2

        return last_node, err_node1, None


    def restore(self, tree_root):
        if tree_node is None:
            raise ValueError("Tree root is None")
        last_node, err_node1, err_node2 = self.inorderTraverse(tree_root, None, None)

        err_node1.value, err_node2.value = err_node2.value, err_node1.value
        return tree_root
from collections import deque

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.value = val
        self.left == left
        self.right = right

def Serialize(object):
    def serialize(self, root):
        if root is None:
            return ""

        queue = deque()
        vals = []
        queue.append(root)
        while len(queue):
            node = queue.popleft()
            if node is None:
                vals.append("None")
            else:
                vals.append(node.value)
                queue.append(node.left)
                queue.append(node.right)

        return " ".join(vals)

    def desialize(self, ser_str):
        parts = ser_str.split()
        if len(parts) == 0:
            return None

        if parts[0] == "None":
            return None

        root = Node(float(parts[0]))
        index = 1
        queue = deque()
        queue.append((root, "left"))
        queue.append((root, "right"))
        while len(queue):
            if index == len(parts):
                break
            if parts[index] == "None":
                node = None
            else:
                node = Node(float(parts[index]))
            top_elem = queue.popleft()
            setattr(top_elem[0], top_elem[1], node)
            if node is not None:
                queue.append((node, "left"))
                queue.append((node, "right"))
            index += 1

        return root

    def dfsSerialize(self, root):
        ser_str = ""
        if root is None:
            return "None"
        stack = []
        stack.append(root)
        while len(stack):
            nd = stack.pop()  # pop from back
            if nd is None:
                ser_str.append("None,")
                continue
            ser_str.append(str(nd.value) + ',')
            stack.append(nd.left)
            stack.append(nd.right)

        return ser_str

    def dfsDerialize(self, ser_str):
        str_parts = ser_str.split(',')
        if len(str_parts) == 0:
            return None
        index = 0
        root_node = Node(float(ser_str[index]))
        index += 1
        stack = [(root_node, "right"), (root_node, "left")]
        while len(stack):
            if index >= len(str_parts):
                break
            nd_ptr = stack.pop()
            nd_val = str_parts[index]
            nd = None
            if nd_val is not None:
                nd = Node(float(nd_val))
                stack.append((nd, "right"))
                stack.append((nd, "left"))
            setattr(nd_ptr[0], nd_ptr[1], nd)
            index += 1

        return root_node

# Given a list of numbers, rearrage it so that no element is equal to the average of its neighbors

class NotEqualToNbrAverage(object):
    def rearrange(self, numList):
        numList.sort()
        result = []
        left, right = 0, len(numList) - 1
        while left <= right:
            result.append(numList[left])
            result.append(numList[right])
            left += 1
            right -= 1

        for i in range(1, len(result)-1):
            if result[i] == (result[i-1] + result[i+1])/2:
                return []
        return result        


# given words in lexicographic order, find the alphabetic order

from typing import List
from collections import defaultdict

class Solution(object):
    def dfs(self, ch, edges, result, visited):
        if ch in visited:
            return visited[ch]
        visited[ch] = True
        for nbr in edges[ch]:
            cycle = self.dfs(nbr, edges, result, visited)
            if cycle:
                return True
        result.append(ch)
        visited[ch] = False
        return False

    def alienDictionary(self, words: List[str]) -> List[str]:
        if len(words) < 2:
            return []

        result = []
        edges = defaultdict(set)

        for i in range(1, len(words)):
            minLen = min(len(word[i-1], word[i]))
            if (word[i-1][0:minLen] == word[i][0:minLen]):
                if (len(word[i]) < len(word[i-1])):
                    return []
                continue
            for j in range(minLen):
                if word[i-1][j] != word[i][j]:
                    edges[word[i-1][j]].add(word[i][j])
                    break

        visited = {}
        for ch in edges.keys():
            cycle = self.dfs(ch, edges, result, visited)
            if cycle:
                return []

        return reversed(result)
# construct all binary trees using a list of numbers

from collections import Counter
import copy

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class AllBinaryTrees(object):
    def recursiveConstruct(self, num_dict, num_nodes):
        if num_nodes == 0:
            res = [None]
        res = []
        for val, nums in num_dict.items():
            if nums == 0:
                continue
            nd = Node(val)
            num_dict[val] -= 1
            for j in range(num_nodes):
                left_trees = self.recursiveConstruct(num_dict, j)
                right_trees = self.recursiveConstruct(num_dict, num_nodes-1-j)
                for lt in left_trees:
                    for rt in right_trees:
                        nd.left = lt
                        nd.right = rt
                        res.append(copy.deepcopy(nd))
            num_dict[val] += 1
        return res


    def construct(self, nums):
        num_dict = Counter(nums)
        trees = self.recursiveConstruct(num_dict, len(nums))
        return trees
        
import numpy as np
import bisect
'''
Given two sorted arrays A and B, generate all possible arrays such that first element is taken from A then from B then from A and so on in increasing order till the arrays exhausted. The generated arrays should end with an element from B.
'''

class SortWays(object):
    def count(self, arrA, arrB, begA, begB):
        if len(arrB) - begB <= 0:
            return 0
        if len(arrA) - begA <= 0:
            return 0
        cnt = 0
        for i in range(begA, len(arrA)):
            a = arrA[i]
            next_b = bisect.bisect_left(arrB, a, low=begB)
            if next_b == len(arrB):
                return 0
            for j in range(next_b, len(arrB)):
                next_a = bisect.bisect_left(arrA, arrB[j], low=i+1)
                if next_a == len(arrA):
                    break
                cnt += self.count(arrA, arrB, next_a, next_b)
        return cnt


    def ways(self, arrA, arrB):
        return self.count(arrA, arrB, 0, 0)

import numpy as np

'''
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete at most two transactions.

Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).

 

Example 1:

Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
Example 2:

Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.
Example 3:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
Example 4:

Input: prices = [1]
Output: 0
'''

class Solution(object):
    def maxProfit1Transaction(self, prices):
        m1Table = np.zeros((len(prices), 3), dtype=np.float)
        min_price = prices[0]
        min_index = 0
        for i in range(1, len(prices)):
            last_sell = m1Table[i-1, 1]
            if min_price > prices[i]:
                min_price = prices[i]
                min_index = i
            if last_sell > 0:
                if prices[last_sell] < prices[i]:
                    m1Table[i, :] = (m1Table[i-1,0] + prices[i]-prices[last_sell], i, m1Table[i-1,2])
                else:
                    m1Table[i,:] = m1Table[i-1,:]
            else:
                if prices[i] > min_price:
                    m1Table[i,:] = (prices[i] - min_price, i, min_index)

        return m1Table

    def maxProfit2Transactions(self, prices, m1Table):
        m2Table = np.zeros(len(prices), dtype=np.float)
        for i in range(1, len(prices)):
            last_buy = m1Table[i,2]
            if last_buy > 1:
                m2Table[i] = max(m2Table[i-1], m1Table[i,0] + m1Table[last_buy-1,0])
        return m2Table

    def maxProfit(self, prices):
        if not prices:
            return 0
        # m2Table[i] = (max profit) 2 trx
        # m1Table[i] = (max_profit, last sell, last_buy) 1 trx
        # m2Table[i] = max(m2Table[i-1], m2Table[i-1] + prices[i] - prices[last_sell])
        m1Table = self.maxProfit1Transaction(prices)
        m2Table = self.maxProfit2Transactions(prices, m1Table)
        return max(0, max(m1Table[-1,0], m2Table[-1]))

class Simple(object):
    def solve1Trx(self, prices):
        max_diff = 0
        min_index = 0
        table = np.zeros(len(prices), dtype=np.float32)
        if len(prices) == 1:
            return 0, 0, 0

        for i in range(1, len(prices)):
            if prices[i] < prices[min_index]:
                min_index = i
            table[i] = prices[i] - prices[min_index]

        max_index = np.argmax(table)
        return max_index, min_index, table[max_index]

    def solve2Trx(self, prices):
        max_profit = 0
        for i in range(1, len(prices)-1):
            tr1 = self.solve1Trx(prices[0:i])
            tr2 = self.solve1Trx(prices[i:])
            max_profit = max(max_profit, tr1[2] + tr2[2])

        return max_profit

'''
Given the root of a binary tree, return the maximum path sum.

For this problem, a path is defined as any node sequence from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

 

Example 1:


Input: root = [1,2,3]
Output: 6
Example 2:


Input: root = [-10,9,20,null,null,15,7]
Output: 42
'''

# S[i] : sum of max path starting at root i and proceeding downwards
#      = max(node[i], node[i] + S[r], node[i] + S[l])
# R[i] = max path running through i
#      = max(node[l] + S[i] if S[i] picks right, node[r] + S[i] otherwise,
#            S[l] + S[i] if S[i] picks right,
#            S[r] + S[i] if S[i] picks left branch
# S[i] = (sum, next_node)
# ans = max(R[])
# write an interator class for binary search tree
# hasNext: bool o(1) amortized, O(h) worst
# next: return val, o(1) amortized
# prev
# O(h) memory, not O(N)

# if O(N), sort

class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BSTIterator(object):
    def __init__(self, root):
        self.stack = []
        self.pushLeftChildren(root)

    def pushLeftChildren(self, root):
        node = root
        while node:
            self.stack.append(node)
            node = node.left

    def hasNext(self):
        return len(self.stack) > 0

    def next(self):
        nd = self.stack.pop()
        result = nd.val
        if nd.right:
            self.pushLeftChildren(nd.right)
        return result

    def value(self):
        return self.stack[-1].val


# given a string of 0 or 1 characters, find the minimum cost to flip
# it to alternating 0, 1 bits: 010101.. or 101010...
# cost of flipping 1 bit: 1
# cost of rotating one bit from left to right: 0

class MistCostAlternateBit(object):
    def findMinCost(self, bitstr):
        result = len(bitstr)
        diff1, diff2 = 0, 0
        sz = len(bitstr)
        newstr = bitstr + bitstr
        patterns = ["01" * sz, "10" * sz]
        left, right = 0, 0
        while right < len(newstr):
            if newstr[right] != patterns[0][right]:
                diff1 += 1
            if newstr[right] != patterns[1][right]:
                diff2 += 1

            curr_len = right - left + 1
            if curr_len > sz:
                if newstr[left] != pattern[0][left]:
                    diff1 -= 1
                if newstr[left] != pattern[1][left]:
                    diff2 -= 1
                left += 1

            if curr_len == sz:
                result = min(result, diff1, diff2)
            
        return result


# reverse the bits of a 32 bit integer and return the int

class Solution(object):
    def highestBit(self, num):
        res = 0
        while num:
            num = num & (num-1)
            res += 1
        return res
    
    def reverseBits(self, num: int) -> int:
        result = 0
        nbits = self.highestBit(num)
        for i in range(nbits):
            result = result | ((num & 1) << (nbits - i))
            num = num >> 1

        return result
import numpy as np

# find bridges and articulation points in a graph
class Graph(object):
    def __init__(self, nvert):
        self.nVert = nvert
        self.edges = []
        for i in range(nvert):
            self.edges.append(set())

    def addEdge(self, u, v):
        self.edges[u].add(v)
        # self.edges[v].add(u)  # for undirected graph

class BridgeArtPoint(object):
    def dfs(self, i, graph, disc_time, low_val, visited):
        if visited[i]:
            return np.inf

        visited[i] = True
        disc_time[i] = i
        low_val[i] = i
        lval = i
        for u in graph.edges[i]:
            lval_child = self.dfs(u, graph, disc_time, low_val, visited)
            if lval_child < lval:
                lval = lval_child
        return lval


    def findBridges(self, graph):
        disc_time = np.full(graph.nVert, -1, dtype=np.int)
        low_val = np.full(graph.nVert, -1, dtype=np.int)
        visited = np.zeros(graph.nVert, dtype=bool)
        for i in range(graph.nVert):
            self.dfs(i, graph, disc_time, low_val, visited)

        bridges = set()
        for u in range(graph.nVert):
            for v in graph.edges[u]:
                if low_val[v] > disc_time[u]:
                    bridges.add((u,v))

        return bridges

import numpy as np
import enum

@enum.unique
class Token(enum.Enum):
    LP = '('
    RP = ')'
    SEMICOL = ';'
    MUL = '*'
    DIV = '/'
    PLUS = '+'
    MINUS = '-'
    VAR = 'VAR'
    NUMBER = 'NUMBER'
    EQUALS = '='


# prog: expr;prog | expr
# expr: term+expr | term - expr | term
# term: prim * term | prim/term | prim
#prim: var | val | var = expr | -prim | (expr)

class Calculator(object):
    # grammar: (); var, val, +, -, *, /
    def __init__(self):
        self.table = {}
        self.tokenTbl = {m.value: m.name for m in list(Token)}
        self.number = None
        self.variable = None
        self.lastVal = None
        self.pos = None
        self.whiteSpace = {' ', '\t', '\n'}


    def getToken(self, line): # reurns token and incremnts pos
        if self.pos >= len(line):
            return None
        ch = line[self.pos]
        while ch in self.whiteSpace:
            self.pos += 1
            if pos >= len(line):
                return None
            ch = line[self.pos]
        
        self.pos += 1
        if ch in self.tokenTbl:
            return self.tokenTbl[ch]

        try:
            val = int(ch)
            self.number = self.parseNumber(line)
            return Token.NUMBER
        except Excption as e:
            self.variable = self.parseVariable(line)
            return Token.VARIABLE

    def prog(self, line):
        if len(line) <= self.pos:
            return
        self.lastVal = self.expr(line)
        if self.getToken == Token.SEMICOL:
            self.prog(line)

    def expr(self, line):
        if len(line) <= self.pos:
            return None
        lhs = self.term(line)
        token = self.getToken(line)
        if token == Token.PLUS:
            return lhs + self.expr(line)
        elif token == Token.MINUS:
            return lhs - self.expr(line)
        elif token is None:
            return lhs
        raise ValueError("Incorrect grammar: %s" % line[self.pos:])

    def term(self, line):
        if self.pos >= len(line):
            return None
        lhs = self.primary(line)
        token = self.getToken(line)
        if token == Token.MULTIPLY:
            return lhs * self.term(line)
        elif token == Token.DIVIDE:
            val = self.term(line)
            if val == 0:
                raise ValueError("Divide by 0")
            return lhs/val
        elif token is None:
            return lhs
        raise ValueError("Incorrect grammar: %s" % line[self.pos:])

    def primary(self, line):
        if self.pos >= len(line):
            return None
        token = self.getToken(line)
        if token == Token.VAR:
            nexttok = self.getToken(line)
            if nexttok is None:
                if self.variable not in self.table:
                    raise ValueError("Var not defined: %s" % self.variable)
                return self.table[self.variable]
            elif nexttok == Token.EQUALS:
                var = self.variable
                val = self.expr(line)
                self.table[var] = val
                return val
        elif token == Token.NUMBER:
            return self.number
        elif token == Token.MINUS:
            return -self.primary(line)
        elif token == Token.LP:
            val = self.expr(line)
            nexttok = self.getToken(line)
            assert nexttok == token.RP
            return val
        return ValueError("Incorrect grammar: %s" % line[self.pos:])

    def calculate(self, line):
        self.pos = 0
        return self.prog(line)

# In a 1 lane way, cars with initial position and speed are specified. for a given ending location, find the number of distinct car fleets that arrive

class CarFleet(object):
    def count(self, pos_arr, speed_arr, loc):
        order_arr = range(len(pos_arr))
        order_arr = sorted(order_arr, key=lambda x: (pos_arr[x], speed_arr[x]))
        end = len(order_arr) - 1
        while end >= 0:
            if pos_arr[end] <= loc:
                break

        time_arr = [ (loc - pos_arr[order_arr[i]])/speed_arr[order_arr[i]] for i in range(end+1) ]
        stack = []
        for i in range(end+1):
            while stack and (time_arr[i] > time_arr[stack[-1]]):
                stack.pop()
            stack.append(i)
        return len(stack)
# get the number of distinct ways we can sum to target. Use nums multiple times

class CSum(object):
    def getSums(self, nums, target):
        sum_dict = {}
        self.dfs(nums, taget, sum_dict)
        return sum_dict.get(target, [])

    def dfs(self, nums, target, sum_dict):
        if target in sum_dict:
            return sum_dict[target]

        if target == 0:
            return []

        if target < 0:
            return None

        result = []
        for nm in nums:
            sums = self.dfs(nums, target - nm, sum_dict)
            if sums is not None:
                for sm in sums:
                    result.append(sm + [nm])

        sum_dict[target] = result
        return result
# given an array of non-negative integers and a target. Return all unique
# combinations of numbers from array using each number once

# counting all combinations: use DP
# returning all combination 2**len(arr)
class Solution(object):
    def recursiveFind(self, arr, target, begin, res, partial_result=[]):
        if target < 0:
            return
        if target == 0:
            if partial_result:
                res.append(partial_result.copy())
            return
        if begin >= len(arr):
            return

         
        # include arr[begin] in sum
        partial_result.append(arr[begin])
        self.resursiveFind(arr, target-arr[begin], begin+1, res, partial_result)
        partial_result.pop()

        # do not include arr[i]
        ptr = begin+1
        while (arr[ptr] == arr[begin]) and (ptr < len(arr)):
            ptr += 1
        self.resursiveFind(arr, target, ptr, res, partial_result)


    def findAllSums(self, arr, target):
        if target < 0:
            return []

        if target == 0:
            if 0 in arr:
                return [0]
            return []

        res = []
        arr = sorted(arr)
        self.recursiveFind(arr, target, 0, res)
        return res

    def recursiveCount(self, arr, target, index, count_dict):
        if (target < 0) or (index >= len(arr)):
            return 0
        if target == 0:
            return 1

        key = (index, target)
        if key in count_dict:
            return count_dict[key]

        include_index = self.recursiveCount(arr, target-arr[index], index+1, count_dict)
        next_ind = index+1
        while (next_ind < len(arr)) and (arr[next_ind] == arr[index]):
            next_ind += 1
        without_index = self.recursiveCount(arr, target, next_ind, count_dict)
        count_dict[key] = include_index + without_index
        return count_dict[key]

    def countWays(self, arr, target):
        count_dict = {} # (index_from, target) -> # ways to sum
        arr = sorted(arr)

        return self.recursiveCount(arr, target, 0, count_dict)
        
# generate all k letter combinations from [1, 2, ... n]

class Solution(object):
    def recursiveGen(self, N, K, begin, result, curr_result):
        if len(curr_result) == K:
            result.append(curr_result.copy())
            return

        for i in range(begin, N):
            curr_result.append(i)
            self.recursiveGen(N, K, i+1, result, curr_result)
            curr_result.pop()


    def generate(self, N, K):
        result = []
        curr_result = []
        self.recursiveGen(N, K, 0, result, curr_result)
        return result
import numpy as np

# for each element, find the max length of subarray with that elem as max elem

# B[i] = contig subarrays ending at i with max elem i
#      = 1 + B[i-1](if A[i] >= A[i-1]) + B[i-1-B[i-1]](if A[i-1-B[i-1]] <= A[i]) +...
# F[i] = forward
# =  1 + F[i+1]I(if A[i] >= A[i+1]) + F[i+1+F[i+1]]I(A[i] >= A[i+1+F[i+1]]) + ...

def maxContigSubArr(arr):
    result = np.ones(len(arr), dtype=np.int))
    if not arr:
        return result
    f_table = np.ones(len(arr), dtype=np.int)
    b_table = np.ones(len(arr), dtype=np.int)

    for i in range(1, len(arr)):
        j = i-1
        while j >= 0:
            if arr[i] >= arr[j]:
                b_table[i] += b_table[j]
            else:
                break
            j -= b_table[j]

    for i in range(len(arr)-2, -1, -1):
        j = i+1
        while j < len(arr):
            if arr[i] >= arr[j]:
                f_table[i] += f_table[j]
            else:
                break
            j += f_table[j]

    result = np.subtract(np.add(b_table, f_table), 1)
    return result
# grid with 0 and 1 given. Count the number of islands

# edges are water 0: water, 1: land

from typing import List

class Solution(object):
    def dfs(self, row, col, matrix, visited, nrow, ncol):
        visited.add((row, col))

        for dr in [-1, 1]:
            for dc in [-1, 1]:
                i = row + dr
                j = col + dc
                if (i < 0) or (i == nrow) or (j < 0) or (j == ncol):
                    continue
                if (matrix[i][j] == "0") or ((i,j) in visited):
                    continue
                self.dfs(i, j, matrix, visited, nrow, ncol)

    def numIslands(self, grid: List[List[str]]) -> int:
        if len(grid) == 0 or len(grid[0]) == 0:
            return 0
        nrow, ncol = len(grid), len(grid[0])
        visited = {}
        nisland = 0
        for i in range(nrow):
            for j in range(ncol):
                if (i,j) not in visited:
                    self.dfs(i, j, matrix, visited)
                    nisland += 1

        return nisland
# in a board with black or whte squares, count the number of black squares
# Only count the largest black square at each location

class CountSquares(object):
    def count(self, board: List[List[int]]) -> int:
        result = 0
        nrow, ncol = len(board), len(board[0])
        sq_count = np.zeros((len(board), len(board[0])), dtype=np.int32)
        for j in range(ncol):
            if board[0][j] == 1:
                sq_count[0, j] = 1
        for i in range(1, nrow):
            if board[i][ncol-1] == 1:
                sq_count[i, ncol-1] = 1

        for i in range(1, nrow):
            for j in range(ncol-2, -1, -1):
                if board[i][j] == 1:
                    minside = min(sq_count[i-1, j], sq_count[i, j+1])
                    sq_count[i, j] = 1 + minside

        
import numpy as np
from collections import NamedTuple

Query = NamedTuple("Query", ('action', 'index'))

def answerQueries(query_arr, N):
    table = np.zeros(N, dtype=np.bool)
    result = []
    for query in query_arr:
        if query.action == 1:
            table[query.index-1] = True
        else:
            res = -1
            for j in range(query.index-1, N):
                if table[j]:
                    res = j
                    break
            result.append(res)

    return result

from sortedcontainers import SortedList

def answerQueries(query_arr, N):
    table = np.zeros(N, dtype=bool)
    result = []
    sset = SortedList()
    for query in query_arr:
        if query.action == 1:
            table[query.index-1] = True
            sset.add(query_index-1)
        else:
            pos = sset.bisect_left(query.index - 1)
            if pos == len(sset):
                result.append(-1)
            else:
                result.append(pos)
    return result
import numpy as np
from collections import defaultdict

# count the number of distinct triangles

class Triangle(object):
    def __init__(self, a=None, b=None, c=None):
        self.a = a
        self.b = b
        self.c = c

    def __hash__(self):
        val = 0
        # itertools.permutations([self.a, self.b, self.c], 3)
        sides = [self.a, self.b, self.c]
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                for k in range(3):
                    if (i == k) or (j == k):
                        continue
                    val += hash((sides[i], sides[j], sides[k]))

        return val

    def __eq__(self, other):
        return set([self.a, self.b, self.c]) == set([other.a, other.b, other.c]))


# a + b + c + a**0.5 + b**0.5 + c**0.5 + a**(1/3) + ...

def countDistinctTriangles(trg_arr):
    tr_dict = defaultdict(0)
    
    for triangle in tr_arr:
        tr_dict[tr_arr] += 1

    return len(tr_dict.keys())
# given a list of course dependencies (prereq)

def depList(graph: List[List[int]]) -> List[int]:
    visited = set()
    visiting = set()
    result = []
    for i, nbr_list in enumerate(graph):
        if i in visited:
            continue
        dfs(i, graph, visiting, visited, result)
    return result

def dfs(vtx, graph, visiting, visited, result):
    if vtx in visited:
        return
    if vtx in visiting:
        result = []
        return
    visiting.add(vtx)
    for nd in graph[vtx]:
        dfs(nd, graph, visiting, visited, result)
    visiting.removevtx)
    visited.add(vtx)
    result.append(vtx)

import numpy as np

# decode a string: a[cd2[d4[f]]] = acddffffdffff

class Decode(object):
    def __init__(self):
        self.end = 0

    def decode(self, codestr, times=1, begin=0):
        res = ""
        if begin >= len(codestr):
            self.end = begin
            return res

        i = begin
        prevNum = 0
        numlist = [str(i) for i in range(10)]
        while i < len(codestr):
            if codestr[i] == '[':
                res += self.decode(codestr, prevNum, i+1)
                i = self.end
            elif codestr[i] == ']':
                self.end = i+1
                return res * times
            elif codestr[i] in numlist:
                prevNum = int(codestr[i])
                i += 1
            else:
                res += codestr[i]
                i += 1

        return res * times

    def decodeStack(self, codestr):
        stack = []
        numlist = [str(i) for i in range(10)]
        for ch in codestr:
            if ch == ']':
                innerVal = ''
                while stack[-1] != '[':
                    innerVal = stack.pop() + innerVal
                stack.pop()
                numStr = ""
                while stack and stack[-1].isdigit():
                    numStr = stack.pop() + numStr
                num = int(numStr)
                innerVal = innerVal * num
                stack.append(innerVal)
            else:
                stack.append(ch)

        return "".join(stack)
# deep copy a list with random pointers

class Node(object):
    def __init__(self, val, nxt=None, random=None):
        self.val = val
        self.next = nxt
        self.random = random

class CopySList(object):
    def copy(self, lst):
        dct = {}
        dct[None] = None
        node = lst
        while node:
            cnode = Node(node.val)
            dct[node] = cnode
            node = node.next

        for nd, cnd in dct.items():
            cnd.next = dct[nd.next]
            cnd.random = dct[nd.random]

        return dct[lst]
# design a data structure to do the following operations
# post(follower_id, post_id)
# follow(follower_id, followee_id)
# unfollow(follower_id, followee_id)
# getFeed(follower_id, limit) -> limit number of most recent tweets from
# list of posts from followees and self, return a list of post ids

from collections import defaultdict
import heapq

class Twitter(object):
    def __init__(self):
        self.followerDict = defaultdict(set)
        self.tweetDict = defaultdict(list)
        self.count = 0

    def post(self, follower_id, post_id):
        self.tweetDict[follower_id].append((self.count, post_id))
        self.count += 1

    def follow(self, follower_id, followee_id):
        self.followerDict[follower_id].add(followee_id)

    def unfollow(self, follower_id, followee_id):
        if follower_id in self.followerDict:
            if followee_id in self.followerDict[follower_id]:
                self.followerDict[follower_id].remove(followee_id)

    def getFeed(self, follower_id, limit):
        result = []
        heap = []
        heapq.heappush(heap, self.tweetDict[follower_id] + (0,follower_id))
        for i, fid in enumerate(self.followerDict[follower_id]):
            heapq.heappush(heap, self.tweetDict[fid] + (0,fid))

        for i in range(limit):
            if not heapq:
                break
            elem = heapq.heappop(heap)
            result.append(elem[1])
            fid = elem[3]
            if elem[2] < len(self.tweetDict[fid]) - 1:
                heapq.heappush(heap, self.tweetDict[fid][elem[2]+1] + (elem[2]+1, fid))
        return result
        
        


import numpy as np
import heapdict

# heapdit.heapdict is an indexed priority queue

class Graph(object):
    def __init__(self, nvert):
        self.edges = []
        for i in range(nvert):
            self.edges.append({})
        self.nVert = nvert

    def addEdge(self, u, v, dist):
        self.edges[u][v] = dist

class Dijkstra(object):
    def ssshortestPath(self, graph, source):
        distArr = np.full(graph.nVert, -1, dtype=np.float32)
        visited = np.zeros(graph.nVert, dtype=bool)
        distArr[source] = 0
        pqueue = heapdict.heapdict()
        pqueue[source] = 0
        while len(pqueue):
            u, dist = pqueue.popitem()
            if visited[u]:
                continue
            visited[u] = True
            distArr[u] = dist
            for v, distv in graph.edges[u].items():
                if visited[v]:
                    continue
                if (v in pqueue) and (pqueue[v] > dist + distv):
                    pqueue[v] = dist + distv
                elif v not in pqueue:
                    pqueue[v] = dist + distv
        return distArr

class BellmanFord(object):
    def ssshortestPath(self, graph, source):
        distArr = np.full(graph.nVert, np.inf, dtype=np.float32)
        distArr[source] = 0
        for i in range(graph.nVert - 1):
            for u in range(graph.nVert):
                for v, dist in graph.edges[u].items():
                    if distArr[v] > distArr[u] + dist:
                        distArr[v] = distArr[u] + dist

        for u in range(graph.nVert):
            for v, dist in graph.edges[u].items():
                if distArr[v] > distArr[u] + dist:
                    raise ValueError("Negative cycle")

        return distArr

class FloydWarshall(object):
    def apshortestPath(self, graph):
        distMatrix = np.full((graph.nVert, graph.nVert), np.inf, dtype=np.float32)
        for u in range(graph.nVert):
            distMatrix[u, u] = 0

            for v, dist in graph.edges[u].items():
                distMatrix[u, v] = dist

        for i in range(graph.nVert):
            for j in range(graph.nVert):
                for k in range(graph.nVert):
                    if distMatrix[i, j] > distMatrix[i, k] + distMatrix[k, j]:
                        distMatrix[i, j] = distMatrix[i, k] + distMatrix[k, j]

        return distMatrix

import copy

class Johnson(object):
    def apshortestPath(self, graph):
        newgraph = copy.deepcopy(graph)
        source = newgraph.nVert
        newgraph.edges[source] = {}
        for i in range(newgraph.nVert):
            newgraph.edges[source][i] = 0

        newgraph.nVert += 1
        spaths = BellmanFord().ssshortestPath(newgraph, source)
        
        # reweight edges
        for u in range(source):
            for v, dist in newgraph.edges[u].items():
                newgraph[u][v] = spaths[u] + dist - spaths[v]

        # run Dijkstra
        distMatrix = np.full((source, source), np.inf, dtype=np.float32)
        for i in range(source):
            distMatrix[i, i] = 0

        newgraph.nVert -= 1

        for u in range(source):
            distMatrix[u, :] = Dijkstra().ssshortestPath(newgraph, u)

        for i in range(source):
            for j in range(source):
                distMatrix[i, j] = distMatrix[i, j] + spaths[j] - spaths[i]

        return distMatrix

import numpy as np
from collections import NamedTuple, deque

#Edge = NamedTuple("Edge", ("flow", "capacity", "u", "v"))

class Edge(object):
    def __init__(self, v, capacity, flow=0):
        self.v = v
        self.capacity = capacity
        self.flow = flow
        assert self.capacity >= self.flow

    def __hash__(self):
        return hash(self.v)

    def __eq__(self, other):
        return self.v == other.v


# graph: dict of dict of edges, graph[u] = {v1:edge_to_v1, v2:edge_to_v2, ...}
# represents u->v1, u->v2 edges
# O(EV^2)
class Dinic(object):
    def __init__(self, graph, source, sink):
        self.graph = graph # list of dict
        self.nVert = len(graph)
        self.source = source
        self.sink = sink
        self.graph[self.sink] = {}

    def findLevelGraph(self, level_graph):
        level_graph[:] = -1
        level_graph[self.source] = 0
        queue = deque()
        queue.append(self.source)

        while len(queue):
            vtx = queue.popleft()
            if level_graph[vtx] == -1:
                for v in self.graph[vtx]:
                    edge = self.graph[vtx][v]
                    if (level_graph[edge.v] == -1) and (edge.flow < edge.capacity):
                        level_graph[edge.v] = level_graph[vtx] + 1
                        queue.append(edge.v)

        return level_graph[self.sink] != -1

    def findFlowOnPath(self, parent):
        v = self.sink
        u = parent[v]
        edge = self.graph[u][v]
        min_flow = edge.capacity - edge.flow
        while v != self.source:
            v = u
            u = parent[u]
            edge = self.graph[u][v]
            if min_flow > edge.capacity - edge.flow:
                min_flow = edge.capacity - edge.flow
        return min_flow

    def pushFlow(self, flow, parent):
        v = self.sink
        u = self.parent[v]
        while v != self.source:
            edge = self.graph[u][v]
            edge.flow += flow
            if u not in self.graph[v]:
                self.graph[v][u] = Edge(u, 0, edge.flow)
            else:
                self.graph[v][u].capacity = edge.flow


    def DFS(self, u, parent, level_graph):
        # O(E)
        #stack = [u] # can also use recursion
        if u == self.sink:
            flow = self.findFlowOnPath(parent)
            self.pushFlow(flow, parent)
            return flow

        flow = 0
        for v in self.graph[u]:
            edge = self.graph[u][v]
            if (level_graph[v] == level_graph[u]+1) and (edge.flow < edge.capacity) and (parent[v] == -1):
                parent[v] = u
                flow += self.DFS(v, parent, level_graph)
                parent[v] = -1
        return flow

    def augmentFlow(self, level_graph):
        parent = np.ndarray(self.nVert, dtype=np.int)
        increm_flow = None
        flow = 0
        while (increm_flow is None) or (increm_flow > 0):
            parent[:] = -1
            parent[self.source] = -2
            increm_flow = self.DFS(self.source, parent)
            flow += increm_flow

        return flow

    def maxFlow(self):
        level_graph = np.ndarray(self.nVert, dtype=np.int)
        flow = 0
        while self.findLevelGraph(level_graph):
            flow += self.augmentFlow(level_graph)

        return flow



# number of unique paths from one corner of 2D grid to another, moving down or sideways

import numpy as np

def numUniquePaths(nrow, ncol):
    arr = np.ones(ncol, dtype=np.int32)
    for i in range(nrow-1):
        for j in range(ncol-2, -1, -1):
            arr[j] += arr[j+1]
    return arr[0]

# find the length of longest common substring in two strings

def longestCommonSubstring(str1, str2, begin1=0, begin2=0, table=None):
    if table is None:
        table = np.full((len(str1), len(str2)), -1, dtype=np.int32)

    if begin1 == len(str1) - 1:
        table[begin1, begin2] = 0 if str1[begin1] in str2[begin2:] else 0
        return table[begin1, begin2]

    if begin2 == len(str2) - 1:
        table[begin1, begin2] = 0 if str2[begin2] in str1[begin1:] else 0
        return table[begin1, begin2]

    if str1[begin1] == str2[begin2]:
        table[begin1, begin2] = 1 + longestCommonSubstring(str1, str2, begin1+1, begin2+1, table)
    else:
        val1 = longestCommonSubstring(str1, str2, begin1+1, begin2, table)
        val2 = longestCommonSubstring(str1, str2, begin1, begin2+1, table)
        table[begin1, begin2] = max(val1, val2)
    return table[begin1, begin2]


def longestCommonSubstring2(str1, str2):
    if len(str1) < len(str2):
        str1, str2 = str2, str1
    table = np.zeros((2, len(str2)+1), dtype=np.int32)
    for i in range(len(str1)-1, -1, -1):
        for j in range(len(str2)-1, -1, -1):
            if str1[i] == str2[j]:
                table[0, j] = 1 + table[1, j+1]
            else:
                table[0, j] = max(table[0, j+1], table[1, j])
        table[1, :] = table[0, :]
    return table[0, 0]

# given an array of stock prices and a period of cooldown, find the maximum profit that can be earned.
# No position at beginning or end of the period
# buy or sell 1 stock. Position cannot exceep 1 or become negative.
# Sell must be followed by cooldown period

class Solution(object):
    def recursiveFind(self, prices, begin, isBuy, cooldown, table):
        if begin >= len(prices):
            return 0
        key = (begin, isBuy)
        if key in table:
            return table[key]
        if isBuy:
            buyPnl = -prices[begin] + self.recursiveFind(prices, begin+1, False, cooldown, table)
            holdPnl = self.recursiveFind(prices, begin+cooldown, True, cooldown, table)
            table[key] = max(buyPnl, holdPnl)
        else:
            sellPnl = prices[begin] + self.recursiveFind(prices, begin + 1, True, cooldown, table)
            holdPnl = self.recursiveFind(prices, begin + cooldown, False, cooldown, table)
            table[key] = max(sellPnl, holdPnl)
        return table[key]

    def maxProfit(self, prices, cooldown):
        table = {} # (day index, is buy) key, max profit value
        self.recursiveFind(prices, 0, True, cooldown, table)

        return table[(0, True)]

# Given a target sum and an infinite number of coins from a list, find the number of unique combinations
# that can give the change


class Solution(object):
    def recursiveFind(self, coins, target, begin, sum, table):
        if begin >= len(coins):
            return 0
        if sum > target:
            return 0
        if sum == target:
            return 1
        key = (begin, sum)
        if key in table:
            return table[key]
        table[key] = self.recursiveFind(coins, target, begin+1, sum+coins[begin], table) + self.recursiveFind(coins, target, begin+1, sum, table)
        return table[key]

    def waysToMakeChange(self, coins, target):
        table = {} # key: (i, sum): value: number of ways
        self.recursiveFind(coins, target, 0, 0, table)
        return table.get((0, target), 0)

    def efficientCountWays(self, coins, target):
        table = np.zeros((2, target+1), dtype=np.int32)
        for coin in coins[::-1]:
            for tgt in range(1, target+1):
                remaining = tgt - coin
                val = table[1, tgt]
                if remaining >= 0:
                    val += table[0, remaining]
                table[0, tgt] = val
            table[1, :] = table[0, :]
        return table[0, target]

# Given an array of numbers, find the number of ways that target can be obtained by either adding
# or subtracting each number


class Solution(object):
    def recursiveFind(self, nums, target, begin, currsum, table):
        if begin >= len(nums):
            if currsum == target:
                return 1
            return 0
        key = (begin, currsum)
        if key in table:
            return table[key]
        table[key] = self.recursiveFind(nums, target, begin+1, currsum+nums[begin], table) + self.recursiveFind(nums, target, begin+1, currsum-nums[begin], table)
        return table[key]

    def numWays(self, nums, target):
        table = {} # (i, currsum) -> num ways to get target
        self.recursiveFind(nums, target, 0, 0, table)
        return table[(0, 0)]


# Given 3 strings, find if the 3rd can be formed by interleaving the first 2

class Interleave(object):
    def recursiveFind(self, str1, str2, resStr, begin1, begin2, table):
        key = (begin1, begin2)
        pos = begin1 + begin2
        if begin1 == len(str1):
            table[key] = (resStr[pos:] == str2[begin2:])
        elif begin2 == len(str2):
            table[key] = (resStr[pos:] == str1[begin1:])

        if key in table:
            return table[key]

        if resStr[pos] == str1[begin1]:
            if self.recursiveFind(str1, str2, resStr, begin1+1, begin2, table):
                table[key] = True
                return True
        if resStr[pos] == str2[begin2]:
            if self.recursiveFind(str1, str2, resStr, begin1, begin2+1, table):
                table[key] = True
                return True

        table[key] = False
        return False

    def possible(self, str1, str2, resStr):
        if len(str1) + len(str2) != len(resStr):
            return False

        table = {} # (pos1, pos2) -> True or False
        self.recursiveFind(str1, str2, resStr, 0, 0, table)
        return table[(0, 0)]

    def efficientPossible(self, str1, str2, resStr):
        if len(str2) > len(str1):
            str1, str2 = str2, str1
        len1, len2 = len(str1), len(str2)
        table = np.zeros(len2+1, dtype=np.bool)
        table[len2] = True
        for j in range(len2-1, -1, -1):
            if resStr[len1+j] == str2[j]:
                table[j] = table[j+1]
            else:
                table[j] = False

        for i in range(len1-1, -1, -1):
            if resStr[i+len2] != str1[i]:
                table[len2] = False

            for j in range(len2-1, -1, -1):
                res = False
                if resStr[i+j] == str1[i]:
                    res = res or table[j]
                if resStr[i+j] == str2[j]:
                    res = res or table[j+1]
                table[j] = res

        return table[0]


# In a two dimensional grid, find the length of longest strictly increasing path with only 4 directions of travel
# N, S, E, W

class LongestPath(object):
    def __init__(self):
        self.dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def recursiveFind(self, matrix, row, col, lastval, lenTable, nrow, ncol):
        if (row < 0) or (row == nrow) or (col < 0) or (col == ncol):
            return 0
        if lastval >= matrix[row][col]:
            return 0
        if lenTable[row, col] != 0:
            return lenTable[row, col]

        val = -np.inf
        for direc in self.dirs:
            nextrow, nextcol = row + direc[0], col + direc[1]
            val = max(val, 1 + self.recursiveFind(matrix, nextrow, nextcol, matrix[row][col], lenTable, nrow, ncol))
        lenTable[row, col] = val
        return val

    def findInMatrix(self, matrix):
        nrow, ncol = len(matrix), len(matrix[0])
        lenTable = np.zeros((nrow, ncol), dtype=np.int32)
        maxval = 0
        for i in range(nrow):
            for j in range(ncol):
                self.recursiveFind(matrix, i, j, -np.inf, lenTable, nrow, ncol)
                maxval = max(maxval, lenTable[i, j])
        return maxval

# Count the number of ways to construct a second string from substrings of first string, keeping the
# relative ordering of substrings intact


class NumSubstrings(object):
    def count(self, str1, str2):
        # table[i, j] = (table[i+1, j+1] + table[i+1, j])*(str1[i] == str2[j]) + table[i+1, j]*(str1[i] != str2[j])
        # = (table[i+1, j+1])*(str1[i] == str2[j]) + table[i+1, j]
        table = np.zeros((2, len(str2)), dtype=np.int32)
        if str1[-1] == str2[-1]:
            table[1, len(str2)-1] = 1

        for i in range(len(str1)-2, -1, -1):
            for j in range(len(str2)-1, -1, -1):
                val = 0
                if str1[i] == str2[j]:
                    val = table[1, j+1]
                val += table[1, j]
                table[0, j] = val
            table[1, :] = table[0, :]
        return table[0, 0]

    def recursiveFind(self, str1, str2, beg1, beg2, table):
        if beg1 == len(str1):
            return 0
        if beg2 == len(str2):
            return 1
        key = (beg1, beg2)
        if key in table:
            return table[key]
        val = 0
        if str1[beg1] == str2[beg2]:
            val = self.recursiveFind(str1, str2, beg1+1, beg2+1, table)
        val += self.recursiveFind(str1, str2, beg1+1, beg2, table)
        table[key] = val
        return val

    def anotherCount(self, str1, str2):
        table = {} # (i, j) -> number of ways
        self.recursiveFind(str1, str2, 0, 0, table)
        return table[(0, 0)]
    
# edit distance: find the minimum edit distance between two strings:
# insert a character
# delete a character
# replace a character

# dist(i, j) = dist(i+1, j+1) if str1[i] == str2[j]
# else 1 + min(stst(i+1, j), dist(i, j+1), dist(i+1, j+1))
class EditDistance(object):
    def find(self, str1, str2):
        table = np.zeros((len(str1)+1, len(str2)+1), dtype=np.int32)
        for j in range(len(str2)+1):
            table[len(str1), j] = len(str2) - j
        for i in range(len(str1)+1):
            table[i, len(str2)] = len(str1) - i
        
        for i in range(len(str1)-1, -1, -1):
            for j in range(len(str2)-1, -1, -1):
                if str1[i] == str2[j]:
                    table[i, j] = table[i+1, j+1]
                else:
                    table[i, j] = 1 + min(table[i+1, j], table[i, j+1], table[i+1, j+1])
                    
        return table[0, 0]
    
# find the maximum reward that can be earned by bursting balloons one at a time.
# reward for bursting balloon i = b[i-1] * b[i] * b[i+1]. End balloons only take valid neighbor
# similar to matrix multiplication.
# think how to partition into subproblems

# cost = max_k (C[0, k-1] + C[k+1, N-1] + b[k]) by bursting balloon k at the end
class BurstBalloons(object):
    def recursiveFind(self, balloons, begin, end, table):
        if table[begin, end] != 0:
            return table[begin, end]
        
        if begin > end:
            return 0
        
        val = 0
        for k in range(begin, end+1):
            val2 = self.recursiveFind(balloons, begin, k-1, table) + self.recursiveFind(balloons, k+1, end, table) + balloons[k]
            val = max(val, val2)
        table[begin, end] = val
        return val
        
    def maxReward(self, balloons):
        nballoon = len(balloons)
        table = np.zeros((nballoon, nballoon), dtype=np.int32)
        self.recursiveFind(balloons, 0, nballoon-1, table)
        return table[0, nballoon-1]
    
    
# regex matching: string, pattern containing chars, '.' (matches any char), '*' matches zero or more of previous match
# m[i, j] = true if word[i:] and pattern[j:] match
# = ismatch = w[i] == p[j] or p[j] == '.'
# if not ismatch, m[i, j] = False
# if p[j+1] == '*': m[i, j] = m[i+1, j+2] or m[i+1, j]
# else m[i, j] = m[i+1, j+1]
class RegexMatch(object):
    def recursiveFind(self, word, pattern, begw, begp, table):
        if (begw == len(word)) and (begp == len(pattern)):
            return True
        if begp == len(pattern):
            return False
        if begw == len(word):
            return all([p == '*' for p in pattern[begp:]])
        
        key = (begw, begp)
        if key in table:
            return table[key]
        
        ismatch = ((word[begw] == pattern[begp]) or (pattern[begp] == '.'))
        if not ismatch:
            table[key] = False
            return False
        if ((begp+1) < len(pattern)) and (pattern[begp+1] == '*'):
            table[key] = self.recursiveFind(word, pattern, begw+1, begp+2, table)
            if table[key]:
                return True
            table[key] = self.recursiveFind(word, pattern, begw+1, begp, table)
        else:
            table[key] = self.recursiveFind(word, pattern, begw+1, begp+1, table)
        return table[key]

    def match(self, word, pattern):
        table = {} # (i, j) -> if word[i:] matches pattern[j:]
        self.recursiveFind(word, pattern, 0, 0, table)
        return table[(0, 0)]

# encode and decode an array of strings as a string, with any character possibly ppresent inside the strings
# num_chars|str1num_chars2|...

class EncodeDecode(object):
    def __init__(self, sep='|'):
        self.sep = sep

    def encode(self, strArr):
        result = ""
        for word in strArr:
            result += str(len(word)) + self.sep + word
        return result

    def decode(self, codedStr):
        result = []
        i = 0
        while i < len(codedStr):
            # parse int
            j = i
            while (j < len(codedStr)) and (codedStr[j] != self.sep):
                j += 1
            if j == len(codedStr):
                raise ValueError("separator missing")
            wlen = int(codedStr[i:j])
            begin = j+1
            end = begin + wlen
            if end >= len(codedStr):
                raise ValueError("coded string missing chars")
            result.append(codedStr[begin:end])
            i = end
        return result

# generate permutations

from typing import List

class Permutations(object):
    def generate(self, chars: List[str]) -> List[List[str]]:
        result = []
        nchar = len(chars)
        stack = chars
        used = {ch:False for ch in chars}
        perm = []
        nused = 0
        while len(stack):
            ch = stack.pop()
            if used[ch]:
                continue
            used[ch] = True
            nused += 1
            perm.append(ch)
            if nused == nchar:
                result.append(perm.copy())
                perm = []
                nused = 0
                for ch2 in chars:
                    used[ch2] = False
            else:
                for ch2 in chars:
                    if not used[ch2]:
                        stack.append(ch2)
                        break
        return result

    def generateRec(self, chars, begin=0):
        if begin == len(chars) - 1:
            return [chars[begin]]
        permutations = self.generateRec(chars, begin+1)
        result = []
        ch = chars[begin]
        for perm in permutations:
            for i in range(len(perm)+1):
                result.append(perm[0:i] + [ch] + perm[i:])
        return result


# Given a row of houses with amount of money, fnd the max sum so that no two consecutive houses are selected
# R[i] -> max sum by robbing houses 0 - i
# = max(V[i] + R[i-2], R[i-1])

class RobHouse(object):
    def maxAmount(self, houses):
        max1, max2 = 0, 0
        for house in range(houses):
            maxval = max(house + max2, max1)
            max2 = max1
            max1 = maxval
        return max1

    def housesInCircle(self, houses):
        """ res = max(maxAount(0--1), maxAmount(1-)"""
        return max(self.maxAmount(houses[0:-1]), self.maxAmount(houses[1:]))

    def housesInBinaryTree(self, node, cache=None):
        if node is None:
            return 0
        if cache is None:
            cache = {}
        if node in cache:
            return cache[node]

        v1 = self.housesInBinaryTree(node.left, cache) + self.housesInBinaryTree(node.right, cache)
        v2 = node.val
        if node.left:
            v2 += self.housesInBinaryTree(node.left.left, cache) + self.housesInBinaryTree(node.left.right, cache)
        if node.right:
            v2 += self.housesInBinaryTree(node.right.left, cache) + self.housesInBinaryTree(node.right.right, cache)
        cache[node] = max(v1, v2)
        return cache[node]

# find the number of ways to decode a string of numbers with characters a - z coded as 1 - 26
# W[i] = 0 if s[i] == '0', else W[i+1] (if s[i] in [1,..9]) + W[i+2] if S[i] in [1, 2]

class Solution(object):
    def recursiveFnd(self, codedStr, begin, table):
        if begin >= len(codedStr):
            return 1

        if table[begin] != -1:
            return table[begin]

        try:
            val = int(codedStr[begin])
        except Exception as e:
            raise ValueError("Invalid char")

        table[begin] = 0
        if val == 0:
            return table[begin]

        if begin + 1 < len(codedStr):
            val2 = int(codedStr[begin:begin+2])
            if val2 <= 26:
                table[begin] += self.recursiveFnd(codedStr, begin+2, table)
            table[begin] += self.recursiveFnd(codedStr, begin+1, table)
        else:
            table[begin] = 1
        return table[begin]

    def numWays(self, codedStr):
        table = np.full(len(codedStr), -1, dtype=np.int32)
        self.recursiveFnd(codedStr, 0, table)
        return table[0]

# find if a graph is a valid tree given undirected edges


class ValidTreeGraph(object):
    def dfsCheck(self, adjSet, visited):
        stack = [] # arguments to recursive function need to be pushed to this stack
        stack.append((adjSet.keys()[0], None)) # (node, parent in traversal)
        while len(stack):
            node, parent = stack.pop()
            if node in visited:
                return True
            visited.add(node)
            for nbr in adjSet[node]:
                if nbr == parent:
                    continue
                stack.append((nbr, node))
        return False

    def find(self, edges):
        adjSet = {}
        for edge in edges:
            if edge[0] not in adjSet:
                adjSet[edge[0]] = {}
            if edge[1] not in adjSet:
                adjSet[edge[1]] = {}
            adjSet[edge[0]].add(edge[1])
            adjSet[edge[1]].add(edge[0])

        N = len(adjSet.keys())
        visited = {}
        loop = self.dfsCheck(adjSet, visited)
        return (not loop) and (len(visited) == N)

# find the minimum positive missing integer in an array
class Solution(object):
    def minMissingPosInt(self, nums):
        N = len(nums)
        for i, num in enumerate(nums):
            if num <= 0:
                nums[i] = N+1

        for i in range(len(nums)):
            actualNum = nums[i]
            if actualNum < 0:
                actualNum *= -1
            if actualNum <= N:
                if nums[actualNum-1] > 0:
                    nums[actualNum-1] *= -1

        for i in range(len(nums)):
            if nums[i] > 0:
                return i+1
        return N+1

# design a stack that supports following operations: push, pop, top, getMin all in o(1)

class MinStack(object):
    def __init__(self):
        self.stack = []

    def push(self, item):
        if len(self.stack):
            minval = min(item, self.stack[-1][1])
            self.stack.append((item, minval))
        else:
            self.stack.append((item, item))

    def pop(self):
        return self.stack.pop()[0]

    def top(self):
        return self.stack[-1][0]

    def getMin(self):
        return self.stack[-1][1]


# given an array with 0, 1, 2. Sort it

def sortSimpleArr(numarr):
    freq = np.zeros(3, dtype=np.int32)
    for num in numarr:
        freq[num] += 1

    return [0]*freq[0] + [1]*freq[1] + [2]*freq[2]

def altSort(numarr):
    left, right = 0, len(numarr)-1
    i = 0
    while i <= right:
        if numarr[i] == 0:
            numarr[left], numarr[i] = numarr[i], numarr[left]
            left += 1
            i += 1
        elif numarr[i] == 2:
            numarr[i], numarr[right] = numarr[right], numarr[i]
            right -= 1
        else:
            i += 1
    return numarr

# find if a binary tree is balanced
from collections import deque
from typing import Tuple

class TreeNode(object):
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

class Solution(object):
    def recursiveFind(self, root, cache) -> Tuple[int, bool]: # height of tree, is balanced
        if root is None:
            return 0, True
        if root in cache:
            return cache[root]

        leftbal = self.recursiveFind(root.left, cache)
        rightbal = self.recursiveFind(root.right, cache)
        isbal = True
        if (not leftbal[1]) or (not rightbal[1]):
            isbal = False
        if abs(leftbal[0] - rightbal[0]) > 1:
            isbal = False
        cache[root] = (1 + max(leftbal[0], rightbal[0]), isbal)
        return cache[root]

    def isTreeBalanced(self, root):
        cache = {}
        res = self.recursiveFind(root, cache)
        return res[1]

    # can also be done non recursively
    def nonrecursive(self, root):
        depthDict = {}
        heightDict = {}
        parentDict = {}
        leafSet = set()
        if root is None:
            return True
        stack = [(root, 0)]
        while len(stack):
            node, depth = stack.pop()
            if node.left:
                stack.push((node.left, depth+1))
                parentDict[node.left] = node
            if node.right:
                stack.push((node.right, depth+1))
                parentDict[node.right] = node
            if self.isLeaf(node):
                leafSet.add(node)
            depthDict[node] = depth

        # now BFS from leaves
        queue = deque(list(leafSet))

        height = 0
        while len(queue):
            numchildren = len(queue)
            newnodes = {}
            for i in range(numchildren):
                node = queue.popleft()
                heightDict[node] = height
                newnodes.add(parentDict[node])
            for nd in newnodes:
                queue.append(nd)
            height += 1

        # dfs
        stack = [root]
        heightDict[None] = 0
        while len(stack):
            node = stack.pop()
            if abs(heightDict[node.left] - heightDict[node.right]) > 1:
                return False
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return True

# create balanced binary tree from sorted arr

class Solution(object):
    def recursiveCreate(self, arr, begin, end):
        if begin > end:
            return None
        mid = (begin + end) // 2
        root = TreeNode(arr[mid])
        root.left = self.recursiveCreate(arr, begin, mid-1)
        root.right = self.recursiveCreate(arr, mid+1, end)
        return root
    
    def balTreeSortedArr(self, arr):
        return self.recursiveCreate(arr, 0, len(arr)-1)


#given a two D grid of alphabets, find if a given word can be constructed going in 4 dirs
#T(i,j,k) : is word[k:] present at (i,j) position
# dfs at i,j using a visited set
# O(N*M*L*4)

class Solution(object):
    def dfs(self, grid, word, row, col, begin, visited, table):
        if begin == self.nword:
            return 1 # found
        if (row < 0) or (col < 0) or (row >= self.nrow) or (col >= self.ncol):
            return 0

        if table[row, col, begin] != -1:
            return table[row, col, begin]

        res = 0
        if ((row, col) not in visited) and (grid[row][col] == word[begin]):
            visited.add((row, col))
            res = self.dfs(grid, word, row+1, col, begin+1, visited, table)
            if res == 0:
                res = self.dfs(grid, word, row - 1, col, begin + 1, visited, table)
            if res == 0:
                res = self.dfs(grid, word, row, col+1, begin + 1, visited, table)
            if res == 0:
                res = self.dfs(grid, word, row, col-1, begin + 1, visited, table)
        table[row, col, begin] = res
        return res

    def isPresent(self, grid, word):
        self.nrow, self.ncol, self.nword = len(grid), len(grid[0]), len(word)
        table = np.full((self.nrow, self.ncol, self.nword), -1, dtype=np.int8) # intitially all -1
        visited = set()

        for i in range(self.nrow):
            for j in range(self.ncol):
                self.dfs(grid, word, i, j, 0, visited, table)

        return table.sum() > 0

# move 0s to right side of an array in place, keeping relative order of other elements same
#

class Solution(object):
    def move0ToRightStable(self, nums):
        left, right = 0, 0
        while (left < len(nums)) and (nums[left] != 0):
            left += 1

        right = left + 1
        while right < len(nums):
            if (nums[right] != 0):
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
            right += 1

# tree of life
# given a grid (2D) with cells 1: alive, 0: dead
# any dead cell with 3 or 4 live neighbors becomes alive
# and live cell with 4 or more dead cell neighbors becomes dead
# Do in place
# 0, (new) 2 -> 0
# 0, 3 -> 1
# 1, 4 -> 0
# 1, 5 -> 1
class Solution(object):
    def countAliveDeadNbrs(self, cellState, row, col):
        nalive, ndead = 0, 0
        nrow, ncol = cellState.shape

        for i in range(-1, 2):
            for j in range(-1, 2):
                nr, nc = row + i, col + j
                if (nr < 0) or (nr >= nrow) or (nc < 0) or (nc >= ncol) or ((i == 0) and (j == 0)):
                    continue
                if cellState[nr, nc] in [1, 3, 5]:
                    nalive += 1
                else:
                    ndead += 1
        return nalive, ndead

    def evolve(self, cellState, timesteps):
        mapping = {2: 0, 3: 1, 4: 0, 5: 1}
        for t in range(timesteps):
            for i in range(cellState.shape[0]):
                for j in range(cellState.shape[1]):
                    alive, dead = self.countAliveDeadNbrs(cellState, i, j)
                    if (alive in [3, 4]) and (cellState[i, j] == 0):
                        cellState[i, j] = 3
                    elif (dead >= 4) and (cellState[i, j] == 1):
                        cellState[i, j] = 4
                    elif cellState[i, j] == 0:
                        cellState[i, j] = 2
                    else:
                        cellState[i, j] = 5

            for i in range(cellState.shape[0]):
                for j in range(cellState.shape[1]):
                    cellState[i, j] = mapping[cellState[i, j]]

        return cellState

# KMP algorithm

class KMP(object):
    def fillLongestPrefixSuffixArr(self, pattern, longestPrefSuf):
        i = 1
        lastPS = 0
        while i < len(pattern):
            if pattern[i] == pattern[lastPS]:
                lastPS += 1
            elif lastPS != 0:
                while ((lastPS > 0) and (pattern[i] != pattern[lastPS])):
                    lastPS = longestPrefSuf[lastPS - 1]
            longestPrefSuf[i] = lastPS
            i += 1

    def fillAllPatternPos(self, word, pattern, longestPrefSuf):
        i, j = 0, 0
        res = []
        while i < len(word):
            if j == len(pattern):
                res.append(i - len(pattern))
            else:
                if word[i] == pattern[j]:
                    j += 1
                else:
                    while ((j > 0) and (word[i] != pattern[j])):
                        j = longestPrefSuf[j-1]
            i += 1
        return res

    def find(self, word, pattern):
        if len(word) < len(pattern):
            return []

        longestPrefSuf = np.zeros(len(pattern), dtype=np.int32)
        self.fillLongestPrefixSuffixArr(pattern, longestPrefSuf)
        return self.findAllPatternPos(word, pattern, longestPrefSuf)

# do inorder traversal of a binary search tree using iterative method
class Inorder(object):
    def recursive(self, root, result):
        if root is None:
            return
        self.recursive(root.left, result)
        result.append(root.val)
        self.recursive(root.right, result)

    def iterRec(self, root):
        result = []
        self.recursive(root, result)
        return result

    def iterate(self, root):
        result = []
        stack = []
        node = root
        while (node or len(stack)):
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            result.append(node.val)
            node = node.right
        return result


class Node(object):
    def __init__(self, val, next_node=None):
        self.val = val
        self.nextNode = next_node

class AddTwoLL(object):
    @staticmethod
    def add(list1, list2):
        result = None
        curr_node = None
        carry = 0
        while (list1 is not None) and (list2 is not None):
            add_val = list1.val + list2.val + carry
            carry = add_val // 10
            node_val = add_val % 10
            node = Node(node_val)
            if result is None:
                result = node

            if curr_node is None:
                curr_node = node
            else:
                curr_node.nextNode = node
                curr_node = node

            list1 = list1.nextNode
            list2 = list2.nextNode

        rem_list = None
        if list1 is None:
            rem_list = list2
        else:
            rem_list = list1
        
        while rem_list is not None:
            add_val = rem_list.val + carry
            carry = add_val // 10
            node_val = add_val % 10
            curr_node.nextNode = Node(node_val)
            curr_node = curr_node.nextNode

        return result


from collections import Counter

class SumPositions(object):
    @staticmethod
    def find(list1, target):
        count_dict = Counter(list1)
        result = []
        for num in list1:
            num2 = target - num
            if num2 in count_dict:
                if (num2 == num):
                    if (count_dict[num] > 1):
                        result.append((num, num2))
                else:
                    result.append((num, num2))

        return result

# given a list of integers and a budget, get the frequency of most frequent integer that can be obtained by incrementing any numbers at most budget times

class Solution(object):
    def freqMostFrequent(self, int_list, limit):
        int_list = sorted(int_list)
        left, right = 0, 0
        max_freq = 0
        win_sum = 0
        while right < len(int_list):
            win_sum += int_list[right]
            max_sum = int_list[right] * (right - left + 1)
            
            while max_sum < win_sum + limit:
                right += 1
                if right == len(int_list):
                    break
                win_sum += int_list[right]
                max_sum = int_list[right] * (right - left + 1)

            while (max_sum > winsum + limit) and (left < right):
                win_sum -= int_list[left]
                left -= 1
                max_sum = int_list[right] * (right - left + 1)
                
            max_freq = max(max_freq, right - left + 1)
        return max_freq

# given an array of gas and an array of cost to go from one station to next,
# find the index of station from where one can go to all stations sequantially
# without running out of gas
# stations arranged around a circle

class Solution(object):
    def travel(self, gas: List[int], cost: List[int]) -> int:
        res = 0
        gasleft = 0
        for i in range(len(gas)):
            gasleft += gas[i] - cost[i]
            if gasleft < 0:
                gasleft = 0
                res = i+1
                continue

        if res == len(gas):
            return -1
        return res

# given a number N, generate all permutations using numbers 1, 2, .., N

class Permutation(object):
    def recursiveGenerate(self, begin, current, N, permutations):
        if begin >= N:
            permutations.append(current.copy())
            return

        for i in range(1, N+1):
            current.append(i)
            self.recursiveGenerate(begin+1, current, N, permutations)
            current.pop()

    def generate(self, N):
        permutations = []
        self.recursiveGenerate(0, [], N, permutations)
        return permutations
import numpy as np

class GramSchmidt(object):
    def classic(self, matrix, replace=True):
        result = matrix
        if not replace:
            result = matrix.copy()

        nrow, ncol = matrix.shape
        for j in range(ncol):
            for i in range(j):
                result[:, j] -= np.dot(result[:, j], result[:, i]) * result[:, i]
            result[:, j] /= np.dot(result[:, j], result[:, j])
        return result

    def modified(self, matrix, replace=True):
        result = matrix
        if not replace:
            result = matrix.copy()

        nrow, ncol = matrix.shape
        for j in range(ncol):
            result[:, j] /= np.dot(result[:, j], result[:, j])
            for i in range(j+1, ncol):
                result[:, i] -= np.dot(result[:, j], result[:, i]) * result[:, j]
           
        return result

import numpy as np

class BillionUser(object):
    def getEnd(self, growth_rates):
        end = 1
        fval = self.evalFunc(growth_rates, end)
        while fval < 0:
            end = end*2
            fval = self.evalFunc(growth_rates, end)
        return end

    def evalFunc(self, growth_rates, day):
        sval = sum([rate**day for rate in growth_rates])
        return sval - 1E9

    def bisection(self, growth_rates, begin, end):
        if end - begin < 1E-2:
            return begin

        mid = (begin + end)/2.0
        fval = self.evalFunc(growth_rates, mid)
        if np.abs(fval) < 1:
            return begin
        elif fval > 0:
            return self.bisection(growth_rates, begin, mid)
        else:
            return self.bisection(growth_rates, mid, end)



    def getBillionUserDay(self, growth_rates):
        begin = 0
        end = self.getEnd(growth_rates)
        # solution bracketed between begin, end
        value = self.bisection(growth_rates, begin, end)
        return int(value)
# using a list of cards, return true if one can form consecutive hands of N cards each

import heapq

class HandOfCards(object):
    def isPossible(self, cards, N):
        ncard = len(cards)
        if ncard % N:
            return False

        cardCount = {}
        for card in cards:
            cardCount[card] = 1 + cardCount.get(card, 0)
        uniqueCards = list(cardCount.keys())
        heapq.heapify(uniqueCards)

        while uniqueCards:
            minCard = uniqueCards[0]
            for k in range(N):
                val = minCard + k
                if cardCount.get(val, 0) == 0:
                    return False
                cardCount[val] -= 1
                if cardCount[val] == 0:
                    if uniqueCards[0] != val:
                        return False
                    heapq.heappop(uniqueCards)
        return True

# given a list of non-overlapping intervals sorted by beginning point and a new interval, merge and return a new set of sorted intervals that are non overlapping and include the new interval

class Interval(object):
    def merge(self, intervals, newInterval):
        result = []
        for i in range(len(intervals)):
            if intervals[i][1] < newInterval[0]:
                result.append(intervals[i])
            elif intervals[i][0] > newInterval[1]:
                result.append(newInterval)
                return result + intervals[i:]
            else:
                newInterval = [min(intervals[i][0], newInterval[0]), max(intervals[i][1], newInterval[1])]

        result.append(newInterval)
        return result
# given two linked lists, find the first common node

class Node(object):
    def __init__(self, value, nextnd = None):
        self.value = value
        self.next_node = nextnd

class Solution(object):
    def firstCommonNode(node1, node2):
        increments1, increments2 = 0, 0
        nd1 = node1
        while nd1:
            nd1 = nd1.next_node
            increment1 += 1

        nd2 = node2
        while nd2:
            nd2 = nd2.next_node
            increment2 += 1

        if increment1 > increment2:
            while increment1 > increment2:
                node1 = node1.next_node
                increment1 -= 1
        elif increment2 > increment2:
            while increment2 > increment1:
                node2 = node2.next_node
                increment2 -= 1

        for i in range(increment1, 0, -1):
            if node1 == node2:
                return node1
            node1 = node1.next_node
            node2 = node2.next_node
            
        return None
# Matrix 0: water, 1: land. Only 1 island. Count perimeter

class OneIsland(object):
    def recPerim(self, matrix, row, col, visited, nrow, ncol):
        if (row < 0) or (row >= nrow) or (col < 0) or (col >= ncol) or (matrix[row][col] == 0):
            return 1
        if (row, col) in visited:
            return 0

        visited.add((row, col))
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        perim = 0
        for dr in dirs:
            irow, jcol = row + dr[0], col + dr[1]
            perim += self.recPerim(matrix, irow, jcol, visited, nrow, ncol)
        return perim

    def perimeter(self, matrix):
        if not matrix:
            return 0

        nrow, ncol = len(matrix), len(matrix[0])
        visited = {}
        for i in range(nrow):
            for j in range(ncol):
                if matrix[i][j]:
                    per = self.recPerim(matrix, i, j, visited, nrow, ncol)
                    if per:
                        return per

        return 0

'''
Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left justified and no extra space is inserted between words.

Note:

A word is defined as a character sequence consisting of non-space characters only.
Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
The input array words contains at least one word.
'''

class Solution(object):
    def justifyLine(self, wordsInLine, maxWidth):
        char_count = sum([len(w) for w in wordsInLine])
        spaces = maxWidth - char_count
        spaces_per_word = int(spaces/(len(wordsInLine)-1))
        right_space = spaces - spaces_per_word*(len(wordsInLine)-1)
        space_str = ' '*spaces_per_word
        return space_str.join(wordsInLine) + ' '*right_space

    def justify(self, words, maxWidth):
        lines = []
        words_in_line = []
        chars_in_line = 0
        for word in words:
            if words_in_line:
                chars_in_line += 1 # space
            chars_in_line += len(word)
            if chars_in_line == maxWidth:
                words_in_line.append(word)
                lines.append(' '.join(words_in_line))
                words_in_line = []
                chars_in_line = 0
            elif chars_in_line > maxWidth:
                lines.append(self.justifyLine(words_in_line, maxWidth))
                words_in_line = [word]
                chars_in_line = len(word)
            else:
                words_in_line.append(word)

        if words_in_line:
            lines.append(' '.join(words_in_line))

        return lines

# given a set of operations, perform

from sortedcontainers import SortedDict

class Solution(object):
    def perform


class Permutation(object):
    def reverse(self, arr, begin, end):
        while begin <= end:
            arr[begin], arr[end] = arr[end], arr[begin]
            begin += 1
            end -= 1

    def nextPerm(self, arr):
        begin = len(arr)-2
        while begin >= 0:
            if arr[begin] < arr[begin+1]:
                index = begin+1
                while (index < len(arr)) and (arr[begin] < arr[index]):
                    index += 1
                index -= 1
                arr[begin], arr[index] = arr[index], arr[begin]
                self.reverse(arr, begin+1, len(arr)-1)
                return True
            begin -= 1

        return False

    def kthPermutation(self, n, k):
        arr = list(range(1, n+1))
        for i in range(2, k+1):
            available = self.nextPerm(arr)
            if not available:
                return ""

        return "".join([str(i) for i in arr])

# find the largest int in a sliding window

from collections import deque

class MaxIntSLidingWindow(object):
    def findAll(self, nums, win_len):
        mono_dec_queue = deque()
        res = []
        left, right = 0, 0
        while right < len(nums):
            sz = right - left + 1
            if sz == win_len:
                res.append(mono_dec_queue[0])

            while mono_dec_queue and (mono_dec_queue[-1] < nums[right]):
                mono_dec_queue.pop() # popright
            mono_dec_queue.append(nums[right])
            right += 1
            if sz == win_len:
                if nums[left] == mono_dec_queue[0]:
                    mono_dec_queue.popleft()
                left += 1
        return res

# given a board of squares with color (black/white) or no color, find if a legal move exists from a square in any of 8 dirs such that begin and end square are same color 

class LegalMove(object):
    def exists(self, board, row, col):
        dirs = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        exists = False
        nrow, ncol = len(board), len(board[0])
        path_len = 0
        r, c = row, col
        color = board[row][col]
        if color == ".":  # no color
            return False
        for direc in dirs:
            r, c = row, col
            opp_color_seen = False
            while ((r >= 0) and (r < nrow)) and ((c >= 0) and (c < ncol)):
                if board[r][c] not in ['.', color]:
                    opp_color_seen = True
                path_len += 1
                if (path_len >= 3) and (board[r][c] == color) and opp_color_seen:
                    return True
                r, c = r + direc[0], c + direc[1]


        return False

class Node(object):
    def __init__(self, val, nxt=None):
        self.val = val
        self.next = nxt

class Solution(object):
    def detectLoop(self, begin):
        if not begin:
            return False

        slow = begin
        if not slow.next:
            return False
        fast = slow.next.next
        while fast:
            if fast == slow:
                return True
            slow = slow.next
            if not fast.next:
                return False
            fast = fast.next.next

        return False

import numpy as np
import bisect

class LIS(object):
    def fastFindSimple(self, arr):
        self.lTable = np.ones(len(arr), dtype=int)  # len table
        self.lastElem = np.zeros(len(arr), dtype=int)
        self.lastElem[:] = -1
        self.lastElem[0] = arr[0]
        last_index = 0

        for i in range(1, len(arr)):
            index = bisect.bisect_right(self.lastElem[0:last_index+1], arr[i])
            if index == last_index+1:
                self.lTable[i] = last_index + 1
                self.lastElem[last_index+1] = arr[i]
                last_index += 1
            else:
                self.lTable[i,:] = index
                self.lastElem[index] = arr[i]

        return np.max(self.lTable[:])
    

    def find(self, arr):
        # L(i) : lis ending at arr[i]
        # L(i) = max(1, 1 + L(k) for 0 <= k < i if arr[k] <= arr[i] 
        # keep k in 2nd element
        self.lTable = np.ones(len(arr), dtype=int)
        #self.lTable[:] = -1

        for i in range(1, len(arr)):
            max_val = 0
            index = -1
            for j in range(i):
                if (self.lTable[j] > max_val) and (arr[j] <= arr[i]):
                    max_val = self.lTable[j]
                    index = j

            self.lTable[i] = max(1, max_val)

        return np.max(self.lTable)


class Solution(object):
    def listAllAnagrams(self, word, target):
        result = []
        if len(target) > len(word):
            return result
        wordCount = {}
        targetCount = {}

        for i in range(len(target)):
            wordCount[word[i]] = 1 + wordCount.get(word[i], 0)
            targetCount[target[i]] = 1 + targetCount.get(target[i], 0)

        

        begin = 0
        end = len(target)
        while end < len(word):
            if wordCount == targetCount:
                result.append(begin)

            end += 1
            if end == len(word):
                break
            wordCount[word[end]] = 1 + wordCount.get(word[end], 0)
            wordCount[word[begin]] -= 1
            if wordCount[word[begin]] == 0:
                wordCount.pop(word[begin])

            begin += 1

        return result
import numpy as np

'''
Find the longest substring that contains all unique characters
'''

# L[i] = longest substring ending in w[i]
#      = L[i-1] + 1 if w[i] not in w[i-1-L[i-1]] ... w[i-1]
#      = i - index where w[index] == w[i] if w[i] in  w[i-1-L[i-1]] ... w[i-1]

class LongestSubstr(object):
    def find(self, word):
        pos_dict = {}
        if len(word) == 0:
            return 0
        max_len = 1
        len_table = np.zeros(len(word), dtpe=int)
        len_table[0] = 1
        pos_dict[word[0]] = 0
        for i in range(1, len(word)):
            begin = len_table[i-1] - i
            end = i-1
            if word[i] not in pos_dict:
                len_table[i] = len_table[i-1] + 1
            else:
                pos = pos_dict[word[i]]
                if pos < begin:
                    len_table[i] = len_table[i-1] + 1
                else:
                    len_table[i] = i - pos
            pos_dict[word[i]] = i
            if len_table[i] > max_len:
                max_len = len_table[i]

        return max_len

# in a 2X2 matrix on non-negative integers, find the length of longest,
# strictly increasing path moving vertically or horizontally only

from typing import List
import numpy as np

class Solution(object):
    def getLongestIncPathFrom(self, row, col, matrix, table, nRow, nCol, prev):
        if (row < 0) or (row == nRow):
            return 0
        if (col < 0) or (col == nCol):
            return 0
        if table[row, col] != -1:
            return table[row, col]

        if matrix[row][col] <= prev:
            return 1

        maxval = 0
        for di in [-1, 1]:
            for dj in [-1, 1]:
                i = row + di
                j = col + dj
                maxval = max(maxval, 1 + self.getLongestIncPathFrom(i, j, matrix, table, nRow, nCol, matrix[row][col]))
        table[row, col] = maxval
        return maxval

    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        nRow, nCol = len(matrix), len(matrix[0])
        if (nRow == 0) or (nCol == 0):
            return 0

        table = np.full((nRow, nCol), -1, dtype=np.int32)
        maxval = 0
        for row in range(nRow):
            for col in range(nCol):
                maxval = max(maxval, self.getlongestIncPathFrom(row, col, matrix, table, nRow, nCol, -1))
        return maxval
import numpy as np

class Solution(object):
    def longestPalindrome(self, word):
        assert len(word)

        if len(word) == 1:
            return 1

        maxlen = 1
        nchars = len(word)
        table = np.zeros((nchars, nchars), dtype=np.int32)
        for i in range(nchars):
            table[i, i] = 1
            if (i+1 < nchars) and (word[i+1] == word[i]):
                table[i, i+1] = 2
                maxlen = 2

        for i in range(nchars):
            for j in range(i+1, nchars):
                if word[j] == word[i]:
                    table[i, j] = table[i+1, j-1] + 2
                    maxlen = max(maxlen, table[i, j])

        return maxlen
import numpy as np

'''
Find the longest parenthesis
L[i] = length of longest parenthesis ending in i
     = 0 if a[i] == '('
     = L[i-1] + 2 if a[i] == ')' and a[i-1] == ')' and a[i-1-L[i-1]] == '(' else 0
     = L[i-2] + 2 if a[i] == ')' and a[i-1] == '('
'''

class LongestParenth(object):
    def find(self, arr):
        if len(arr) <= 1:
            return 0

        l_table = np.zeros(len(arr), dtype=int)
        for i in range(1, len(arr)):
            if arr[i] == '(':
                l_table[i] = 0
            else: # arr[i] == ')'
                if arr[i-1] == '(':
                    if i > 1:
                        l_table[i] = l_table[i-2] + 2
                    else:
                        l_table[i] = 2
                else: # arr[i-1] == ')'
                    if arr[i-1-l_table[i-1]] == '(':
                        l_table[i] = l_table[i-1] + 2
                    else:
                        l_table[i] = 0

        return l_table.max()
# given a list of stock closing prices (daily), find the maximum span
# Span on day i is defined as the longest K for which P[i-j] < P[i] for 
# j in range(k)

class Solution(object):
    def longestSpan(self, prices: List[float]) -> int:
        if not prices:
            return 0

        maxspan = 1
        spans = np.ones(len(prices), dtype=np.int32)
        for i in range(1, len(prices)):
            j = i-1
            while (j >= 0) and (prices[j] < prices[i]):
                j = j - spans[j]

            spans[i] = i - j
            maxspan = max(maxspan, spans[i])

        return maxpsan
# parition a string a alphabets into max number of groups so that no two groups share a character

class MaxParition(object):
    def partition(self, astr):
        result = []
        last_pos = {}
        for i, ch in enumerate(astr):
            last_pos[ch] = i

        start, end = 0, 0
        while start < len(astr):
            end = max(end, last_pos[start])
            if start == end:
                result.append(start)
            start += 1


# find a recatangle with max area in histogram of 1 unit width bars

class Solution(object):
    def maxArea(self, heights):
        maxarea = 0
        stack = []
        for i, height in enumerate(heights):
            pos = i
            while stack and stack[-1][1] > height:
                pos, ht = stack.pop()
                height = ht
                maxarea = max(maxarea, height*(i - pos))
            stack.append((pos, height))

        for i in range(len(stack), -1, -1):
            maxarea = max(maxarea, stack[i][1]*(stack[i][0] - len(heights)));
        return maxarea

# Given a list of balana piles, find the minimum number of bananas per hour that must be consumed by Koko to finish the piles before or at specified time
# in one hour only 1 pile can be eaten
# => len(piles) <= limit

import math

class Solution(object):
    def getTime(self, piles, speed):
        nhours = 0
        for pile in piles:
            nhours += math.ceil(pile / speed)
        return nhours
    
    def minSpeed(self, piles, limit):
        speed = limit
        arr = list(range(1, max(piles)+1))
        left, right = 0, len(arr)-1
        while left <= right:
            piv = (left + right) // 2
            nhours = self.getTime(piles, arr[piv])
            if nhours > limit:
                left = pivot + 1
            else:
                speed = min(speed, arr[piv])
                right = piv - 1

        return speed

'''
Given a histogram, find the rectangle with the maximum area
'''

class Solution(object):
    # maxArea[i] = (max_area, width, height) of max rect ending in bar i
    #            = max(bar[i]: (bar[i], 1, bar[i])
    #                  (maxArea[i-1,1] + 1)*bar[i] if bar[i] < maxArea[i-1,2]
    #                  (maxArea[i-1,1] + 1)*maxArea[i-1,2] if bar[i] > maxArea[i-1,2]

    def maxArea(self, bars):
        maxArea = np.zeros((len(bars), 3), dtype=np.int)
        if len(bars) == 0:
            return 0
        maxArea[0,:] = (bars[0],1,bars[0])
        soln_max = bars[0]
        for i in range(1, len(bars)):
            max_val = bars[i]
            maxArea[i,:] = (bars[i], 1, bars[i])
            if maxArea[i-1,2] > bars[i]:
                val = (maxArea[i-1,1] + 1)*bars[i]
                if val > max_val:
                    max_val = val
                    maxArea[i,:] = (max_val, maxArea[i-1]+1, bars[i])
            else:
                val = (maxArea[i-1,1] + 1)*maxArea[i-1,2]
                if val > max_val:
                    max_val = val
                    maxArea[i,:] = (max_val, maxArea[i-1, 1]+1, maxArea[i-1,2])
            if soln_max < max_val:
                soln_max = max_val

        return soln_max
import numpy as np
from sortedcontainers import SortedList

class MaxCandy(object):
    def find(self, arr, k):
        if len(arr) == 0:
            return 0

        slist = SortedList(arr, key=lambda x: -x) # descending

        candy = 0
        for i in range(k):
            item = slist.pop(0)
            candy += item
            slist.add(item/2)

        return candy
# find the maximum characters from  a list that can be removed from a 
# string so that a second string is a sub sequence of the first string

# st1 = "abncdfs" st2 = "nfs" pos_arr = [4,1,6], max chars: 2: [4, 1]

# subseq need not be continuous

class Subseq(object):
    def find(self, str1, str2, removed, begin):
        if begin == len(str2):
            return True
        for indx in range(len(str1)):
            if indx in removed:
                continue
            if str1[indx] == str2[begin]:
                removed.add(indx)
                val = self.find(str1, str2, removed, begin+1)
                #if val:
                #    return True

                #removed.remove(indx)
                return val
        return False

    def findMaxNum(self, str1, str2, positions):
        removed = set()
        maxlen = -1
        left, right = 0, len(positions)-1
        while left <= right:
            mid = (left + right) // 2
            removed = set([positions[i] for i in range(mid)])
            if self.find(str1, str2, removed, 0):
                maxlen = max(maxlen, mid)
                left = mid + 1
            else:
                right = mid - 1
        if maxlen == -1:
            return None
        return maxlen
# given a list of integers, find the length of longest consecutive seq

class LongestConsecSeq(object):
    def findLen(self, arr):
        num_set = set(arr)
        max_len = 0
        for num in arr:
            if (num-1) not in num_set:
                incr = 1
                while num + incr in num_set:
                    incr += 1
                max_len = max(max_len, incr)
        return max_len
from operator import itemgettr, attrgettr

def biggestDiff(elems):
    elems2 = [(i,e) for i,e in enumerate(elems)]
    sorted_elems = sorted(elems2, key=lambda x: x[1])
    # sorted_elems = sorted(elems2, key=itemgettr(1))
    maxDiff = -1
    rightLim = 0

    for i in range(len(sorted_elems)):
        for j in range(len(sorted_elems)-1, rightLim, -1):
            if sorted_elems[i][0] < sorted_elems[j][0]:
                maxDiff = max(maxDiff, sorted_elems[j][1] - sorted_elems[i][1])
                rightLim = j
                break
        if rightLim <= i:
            break    


    return maxDiff if maxDiff != -1 else None

# Given a list of tickets (from, to), i.e. (d1, dj) return a list using all tickets, d1, d2, d3, ...

from typing import List
from collections import defaultdict

class Solution(object):
    def dfs(self, vertex, adjList, visited, visitedList):
        if vertex in visited:
            return

        visited.add(vertex)
        for nbr in adjList[vertex]:
            self.dfs(nbr, adjList, visited, visitedList)
        visitedList.append(vertex)


    def maxItin(self, tickets: List[List(int)]) -> List[int]:
        # build adj list
        adjList = {}
        for ticket in tickets:
            if ticket[0] not in adjList:
                adjList[ticket[0]] = []
            adjList[ticket[0]].append(ticket[1])

        for vertex in asjList.keys():
            visited = set()
            visitedList = []
            self.dfs(vertex, adjList, visited, visitedList)
            if len(visitedList) == len(self.adjList.keys()):
                return visitedList.reverse()

        return []

# Given a team on N players, create a team of K players to maximize performance. Two arrays containing speed and efficiency are given. Efficiency of the team is the lowest efficiency of a team member. Performance = sum of speeds of team members * efficiency

import heapq

class MaxPerfTeam(object):
    def maxPerf(self, speed, eff, K):
        max_perf = 0
        sum_speed = 0

        # sort in descending order by speed
        comb_list = [(s,e) for s,e in zip(speed, eff)]
        comb_list.sort(reversed=True, key=lambda x: x[0])

        heap_eff = []
        for i in range(K):
            sum_speed += comb_list[i][0]
            heapq.heappush(heap_eff, (comb_list[i][1], comb_list[i][0]))

        max_perf = sum_speed * heap_eff[0][0]
        for i in range(K, len(comb_list)):
            low_eff = heapq.heappop(heap_eff)
            sum_speed -= low_eff[1]
            heapq.heappush(heap_eff, (comb_list[i][1], comb_list[i][0]))
            sum_speed += comb_list[i][0]
            max_perf = max(max_perf, sum_speed * heap_eff[0][0])

        return max_perf
'''
Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.
'''

import numpy as np

class Solution(object):
    def maximalRectangle(self, matrix):
        # max_area[i,j] : ending at square (i,j)
        # -> (area, x_dim, longest_1_dim0, longest_1_dim1) calc y_dim
        # len_y: length of 1s starting from i,j
        # len_x: length of 1s along x starting from i,j
        #  = max(max_area[i-1,j,0] + y_dim if y_dim <= len_y
        #        (max_area[i-1,j,1]+1)*len_y if y_dim > len_y
        #        max_area([i,j-1,0] + x_dim if x_dim <= x_len

        if (len(matrix) == 0) or (len(matrix[0]) == 0):
            return 0
        max_area = np.zeros((len(matrix), len(matrix[0]), 4), dtype=np.int)
        if matrix[0][0] == 1:
            max_area[0,0,:] = (1,1,1,1)

        for i in range(1, len(matrix)):
            if matrix[i][0] == 1:
                max_area[i,0,:] = (max_area[i-1,0,0]+1, 1, 1, max_area[i-1,0,0]+1)

        for j in range(1, len(matrix[0])):
            if matrix[0][j] == 1:
                max_area[0,j,:] = (max_area[0,j-1,0]+1, max_area[0,j-1,1]+1, max_area[0,j-1,1]+1,1)

        for i in range(1, len(matrix)):
            for j in range(1, len(matrix[0])):
                if matrix[i][j] == 1:
                    max_area[i,j,2] = 1 + max_area[i-1,j,2]
                    max_area[i,j,3] = 1 + max_area[i,j-1,3]
                    len_x = matrix[i,j,2]
                    len_y = matrix[i,j,3]
                    y_dim = matrix[i-1,j,0]/matrix[i-1,j,1]
                    if len_y >= y_dim:
                        matrix[i,j,0] = matrix[i-1,j,0] + y_dim
                        matrix[i,j,1] = matrix[i-1,j,1] + 1
                    else:
                        matrix[i,j,0] = len_y*(matrix[i-1,j,1]+1)
                        matrix[i,j,1] = matrix[i-1,j,1] + 1

                    len_x = matrix[i,j-1,2]
                    len_y = matrix[i,j-1,3] + 1
                    x_dim = matrix[i,j-1,0]/matrix[i,j-1,1]
                    if x_dim <= len_x:
                        return
                        

class Soln(object):
    def find(self, matrix):
        if len(matrix) == 0:
            return 0

        self.table = np.zeros((matrix.shape[0], matrix.shape[1], 2), dtype=np.float)
        # T[i,j,0] = (T[i-1,j,0]/T[i-1,j,1] + 1)*min(T[i-1,j,1], Ones[i,j,2]) if matrix[i,j] == 1, else 0, 0
        # (area, len_y_area)
        
        # find the length of ones along y 
        self.lenOnesY = matrix.copy()
        for j in range(1, len(matrix.shape[1])):
            for i in range(len(matrix.shape[0])):
                if self.lenOnesY[i,j] != 0:
                    self.lenOnesY[i,j] = 1 + self.lenOnesY[i,j-1]

        for j in range(len(matrix.shape[1])):
            if matrix[0,j] == 0:
                self.table[0,j,:] = (0, 0)
            elif j == 0:
                self.table[0,0,:] = (1,1)
            else:
                self.table[0,j,:] = (self.table[0,j-1,0] + 1, self.table[0,j-1,1]+1)

        for i in range(1, len(matrix.shape[0])):
            for j in range(len(matrix.shape[1])):
                if matrix[i,j] == 0:
                    self.table[i,j,:] = (0,0)
                else:
                    y_dim = min(self.table[i-1,j,1], self.lenOnesY[i,j])
                    if y_dim == 0:
                        self.table[i,j,:] = (1,1)
                    else:
                        x_dim = self.table[i-1,j,0]/self.table[i-1,j,1]
                        self.table[i,j,:] = ((x_dim+1)*y_dim, y_dim)

        return self.table[:, :, 0].max()



import numpy as np
from typing import List
from functools import cmp_to_key

# Given a list of positive integers, find the maximum number that can
# be formed by concatenating them

def compare(int1, int2):
    val1 = int("%d%d%" % (int1, int2))
    val2 = int("%d%d" % (int2, int1))
    return val1 - val2

class Solution(object):
    def maxNum(self, lst: List[int]) -> int:
        slist = sorted(lst, key=cmp_to_key(compare))
        st = "".join([str(v) for v in slist])
        return int(st)

class MaxSum(object):
    def maxSumSubArray(self, arr: list) -> int:
        if not arr:
            raise ValueError("need a list")

        maxsum = arr[0]
        prevsum = 0
        for num in arr:
            if prevsum <= 0:
                prevsum = 0
            prevsum += num
            maxsum = max(maxsum, prevsum)

        return maxsum

import numpy as np
import bisect

class Water(object):
    def fillWater(self, sorted_arr, begin, end, height):
        water = 0
        if begin == 0:
            min_ht = sorted_arr[0][0]
            for i in range(1, end):
                water += (min_ht - sorted_arr[i][0])*sorted_arr[i][1]
            sorted_arr[0][0] = height
            sorted_arr[0][1] = 1
        else:
            for i in range(begin, end):
                water += (height - sorted_arr[i][0])*sorted_arr[i][1]
            sorted_arr[begin][0] = height
            sorted_arr[begin][1] = end - begin
        return water

    def maxWater(self, heights):
        if len(heights) < 3:
            return 0
        begin = 0
        water = 0

        for i in range(1, len(heights)):
            if heights[i] < heights[i-1]:
                break
            begin = i

        sorted_arr = [None] * len(heights)
        sorted_arr[0] = (heights[begin], 1)
        end = 1
        for i in range(begin+1, len(heights)):
            if heights[i] < heights[i-1]:
                sorted_arr[end] = (heights[i], 1)
                end += 1
            elif heights[i] == heights[i-1]:
                sorted_arr[end-1][1] += 1
            else:
                pos = bisect.bisect_left(sorted_arr[0:end], -heights[i], key=lambda x:-x[0])  # descending order in sorted_arr
                water += self.fillWater(sorted_arr, pos, end, heights[i])
                end = pos+1
                

        return water

# given 2 sorted lists, find the median

class Median(object):
    def median(self, a1, b1, a2, b2, isOdd):
        if isOdd:
            return max(a1, b1)
        return (max(a1, b1) + min(a2, b2)) / 2
    
    def sortedLists(self, list1, list2):
        sz = len(list1) + len(list2)
        mindex = sz // 2 - 1
        if len(list1) < len(list2):
            list1, list2 = list2, list1

        # list2 is the shorter list
        left, right = 0, len(list2) - 1
        while left <= right:
            mid2 = (left + right) // 2
            mid1 = mindex - mid2 - 1

            a1, b1 = -np.inf, -np.inf
            a2, b2 = np.inf, np.inf
            if mid1 >= 0:
                a1 = list1[mid1]
            if mid1+1 < len(list1):
                a2 = list1[mid1+1]
            if mid2 >= 0:
                b1 = list2[mid2]
            if (mid2+1) < len(list2):
                b2 = list2[mid2+1]

            if (a1 <= b2) and (a2 >= b1):
                return self.median(a1, b1, a2, b2, sz % 2)
            elif a1 > b2:
                left = mid2 + 1
            else:
                right = mid2 - 1



# given a stream of numbers, find the medians for each subarr starting at 0

import heapq

class Stream(object):
    def __init__(self):
        self.smallHeap = [] # max heap 
        self.largeHeap = [] # min heap
        self.count = 0

    def median(self, newNumber):
        self.count += 1
        while (len(self.smallHeap) > len(self.largeHeap)):
            val = -heapq.heappop(self.smallHeap)
            heapq.heappush(self.largeHeap, val)

        if (len(self.smallHeap)) and (-self.smallHeap[0] > newNumber):
            heapq.heappush(self.smallHeap, -newNumber)
        elif (len(self.largeHeap)) and (self.largeHeap[0] <= newNumber):
            heapq.heappush(self.largeHeap, newNumber)
        else:
            heapq.heappush(self.smallHeap, -newNumber)

        if self.count % 2 == 0:
            return (-self.smallHeap[0] + self.largeHeap[0])/2
        return self.largeHeap[0]

    def meadianStream(self, numStream):
        result = []
        for num in numStream:
            result.append(self.median(num))
        return result



# given an list of 0 (can step) and 1 (cannot step), find minimum number of jumps to reach end from begin if one can take between (min, max) jumps from each valid spot

# N number of spots
# q max - min
# DP solution: N*q
# This solution: N

from collections import deque

class MinJumps(object):
    def findMinJumps(self, spots, min_j, max_j):
        nspot = len(spots)
        visited = np.zeros(nspot, dtype=np.bool)
        if len(spots) < 2:
            return None
        if (spots[0] == '1') or (spots[-1] == '1'):
            return None

        queue = deque([(0,0)])
        
        while queue:
            spot, njump = queue.popleft()
            if visited[spot]:
                continue
            if spot == npot-1:
                return njump
            visited[spot] = True
            if spot+min_j >= nspot:
                return None
            for i in range(spot+min_j, min(spot+max_j+1, nspot)):
                if (spots[i] == '0') and (i not in visited):
                    queue.append((i, njump+1))

        return None
# find the length of minimum continuous subseq that sums to K or > K
class MinLenSubseq(object):
    def find(self, nums, K):
        left, right = 0, 0
        min_len = len(nums) + 1
        win_sum = 0
        for right in range(len(nums)):
            win_sum += nums[right]
            while (win_sum >= K) and (left <= right):
                min_len = min(right - left + 1)
                win_sum -= nums[left]
                left += 1
        return min_len
# given a list of intervals and a list of points, find the shortest 
# interval for each point

# interval: [left, right]
# query: x1

import heapq

# O(nlogn + qlogn + qlogq)

class MinLengthInt(object):
    def findIntervals(self, intervals, queries):
        intervals.sort(key = lambda x: x[0])
        pts = sorted(queries)
        heap = []

        result = {}
        begin = 0
        nint = len(intervals)
        for point in pts:
            while begin < nint:
                if intervals[begin][0] > point:
                    break
                heapq.heappush(heap, (intervals[begin][1] - intervals[begin][0] + 1, intervals[begin][1]))

            while heap and (heap[0][1] < point):
                heapq.heappop(heap)

            res = -1
            if heap:
                res = heap[0][0]
            result[point] = res

        return [result[pt] for pt in queries]
# given a subarray of non-negative integers and a splitting size S, find the minimum size of maximum sum of subarray for performing S splits

import numpy as np

class MinMaxSplit(object):
    def findMaxSubarr(self, arr, splits, from_ind, memo={}):
        key = (from_ind, splits)
        if key in memo:
            return memo[key]
        if splits == 0:
            memo[key] = np.sum(arr[from_ind:])
            return memo[key]

        mval = 0
        for indx in range(from_ind+1, len(arr)):
            mval = max(mval, np.sum(arr[from_ind:indx]), self.findMaxSubarr(arr, splits-1, indx, memo))

        memo[key] = mval
        return mval

    def minMax(self, arr, splits):
        if splits == 0:
            return np.sum(arr)

        
        cumsum = np.cumsum(arr)
        min_val = cumsum[-1]
        max_val_dict = {}
        for begin in range(1, len(arr)):
            min_val = min(min_val, max(cumsum[begin-1], self.findMaxSubarr(arr, splits-1, begin, max_val_dict)))
        return min_val

    def subArrMaxSum(self, arr, splits, max_sum):
        if splits == 0:
            return sum(arr) < max_sum

        sum_vals = 0
        nsplit = 0
        for val in arr:
            sum_vals += val
            if sum_vals > max_sum:
                nsplit += 1
                if nsplit > splits:
                    return False
                sum_vals = val
        return True


    def fastFindMaxSubarr(self, arr, splits):
        all_sum = np.sum(arr)
        if splits == 0:
            return all_sum

        left, right = max(arr), all_sum

        result = right
        while left <= right:
            piv = (left + right) // 2
            if self.subArrMaxSum(arr, splits, piv):
                right = piv - 1
            else:
                left = piv + 1
            result = piv

        return result
# given a matrix from top to bottom, left to right, find the minimum 
# path cost (sum of square values along the path) from top, left to 
# bottom right corner, moving either down or to right (east)

# S(m, n) = min(S(m+1, n), S(m, n+1))
class MinPathSum(object):
    def find(self, matrix):
        nrow, ncol = len(matrix), len(matrix[0])
        rowtable = np.zeros(nrow, dtype=np.int32)
        coltable = np.zeros(ncol, dtype=np.int32)

        for i in range(nrow-2, -1, -1):
            rowtable[i] += rowtable[i+1]

        for j in range(ncol-2, -1, -1):
            coltable[i] += coltable[i+1]

        for i in range(nrow-2, -1, -1):
            coltable[ncol-2] = min(rowtable[i], coltable[ncol-2])
            for j in range(ncol-3, -1, -1):
                coltable[j] = min(coltable[j+1], coltable[j])

        return coltable[0]
# find the minimum number of perfect squares required to sum up to N


import numpy as np

class MinSquares(object):
    def findMinNum(self, num, sumval):
        minNum = np.full(num+1, -1, dtype=np.int32)
        cache = np.full[num, num, dtype=np.int32)
        minNum[0] = 0
        for n in range(num):
            for k in range(1, np.floor(np.sqrt(n))):
                sq = k*k
                remain = sumval - sq
                if remain < 0:
                    break
                minNum[n] = min(1 + minNum[remain], minNum[n])

        return 
'''
Given two strings s and t, return the minimum window in s which will contain all the characters in t. If there is no such window in s that covers all characters in t, return the empty string "".

Note that If there is such a window, it is guaranteed that there will always be only one unique minimum window in s.

 

Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Example 2:

Input: s = "a", t = "a"
Output: "a"

first find a window
then find most compact window by incrementing left end
then increment left end by 1 (or until char in word), and fing the next right pointer to that char that was removed from left 
'''

import os

class Solution(object):
    def getFirstBox(self, word, window, counter):
        right_ptr = 0
        moving_dict = {}
        unique_chars = len(counter.keys())

        cnt = 0
        first = None
        right_window_ptr = None
        while (right_ptr < len(word)) and (cnt < unique_chars):
            ch = word[right_ptr]
            if (ch in counter) and (moving_dict.get(ch, 0) < counter[ch]):
                if first in None:
                    first = right_ptr
                right_window_ptr = right_ptr
                moving_dict[ch] = moving_dict.get(ch, 0) + 1
                if moving_dict[ch] == counter[ch]:
                    cnt += 1
            right_ptr += 1

        if first is None:
            return 0, 0, {}

        if cnt == unique_chars:
            return first, right_window_ptr, moving_dict

        return 0, 0, {}


    def minWindow(self, word, window):
        counter = {}
        keys = 0
        for s in window:
            counter[s] = counter.get(s, 0) + 1
            keys += 1

        left_ptr, right_ptr, moving_dict = self.getFirstBox(word, window, counter)
        min_len = right_ptr - left_ptr + 1

        # try to improve
        while (right_ptr < len(word)) and (left_ptr < right_ptr):
            # increment left
            while word[left_ptr] not in moving_dict:
                left_ptr += 1
            ch = word[left_ptr]
            left_ptr += 1
            while word[left_ptr] not in moving_dict:
                left_ptr += 1

            # increment right
            while (right_ptr < len(word)) and word[right_ptr] != ch:
                right_ptr += 1

            if right_ptr == len(word):
                return min_len

            min_len = min(min_len, right_ptr - left_ptr + 1)

        return min_len





# Given start and end times for a list of meetings, find the minimum number of conference rooms

import heapq

class ConfRooms(object):
    def minrooms(self, meetings: List[List]) -> int:
        maxrm = 0
        meetingHeap = []
        meetings = sorted(meetings, key=lambda x: x[0])
        for meeting in meetings:
            while (len(meetingHeap)) and (meetingHeap[0] < meeting[0]):
                heapq.heappop(meetingHeap)
            heapq.heappush(meetingHeap, meeting[1])
            maxrm = max(maxrm, len(meetingHeap))
        return maxrm


# given a list of integers find the first missing positive integer from 1 - n.

class Soln(object):
    def firstMissing(self, numList):
        N = len(numList)
        # position = num-1
        for i, num in enumerate(numList):
            if (num <= 0) or (num >= N+1):
                continue
            if num == i+1:
                continue
            while num != (i+1):
                if (num <= 0) or (num >= N+1):
                    break
                tmp1, tmp2 = numList[num-1], num-1
                numList[num-1] = num
                num, i = tmp1, tmp2
            
        for i, num in enumerate(numList):
            if (num <= 0) or (num >= N+1) or (num-1 == i):
                continue
            return i+1
        return N+1



            
def findMissing(nums):
    for i, num in enumerate(nums):
        if i == num-1:
            continue
        elif (num <= 0) or (num >= len(nums)):
            continue
        else:
            putInPlace(i, nums)

        for i in range(len(nums)):
            if nums[i] != i+1:
                return i+1
        return len(nums) + 1

def putInPlace(pos, nums):
    newPos = nums[pos] - 1
    while (newPos >= 0) and (newPos  < len(nums)) and (nums[newPos]-1 != newpos):
        tmp = nums[newPos]
        nums[newPos] = newPos + 1
        newPos = tmp - 1

# given a list of numbers. Return the most frequent number in a sliding 
# window. If a tie occurs, return the most recently added element

class MRUFreq(object):
    def find(self, nums, win_len):
        left, right = 0, 0
        res = []
        freq_dict = {}
        counts = {}
        max_freq = 0
        if win_len == 0:
            return res
        while right < len(nums):
            sz = right - left + 1
            if sz == win_len:
                res.append(freq_dict[max_freq][-1])
            rch = nums[right]
            count[rch] = count.get(rch, 0) + 1
            freq = count[rch]
            if freq not in freq_dict:
                freq_dict[freq] = deque()
            freq_dict[freq].append(rch)
            if sz == win_len:
                lch = nums[left]
                freq = count[lch]
                freq_dict[freq].popleft()
                count[lch] -= 1
                left += 1
            max_freq = max(max_freq, count[rch])

            right += 1
        return res



import numpy as np

class Node(object):
    def __init__(self, val):
        self.val = val
        self.children = []

def buildLookup(root, word):
    lookup = {word[root.val-1] : 1}
    for child in root.children:
        if child is not None:
            buildLookup(child, word)
            for k,v in child.lookup.items():
                lookup[k] = lookup.get(k, 0) + v
    root.lookup = lookup

def answerQueries(root, queries, word):
    buildLookup(root, word)
    result = []
    for query in queries:
        node, index = query
        result.append(node.lookup.get(word[index-1], 0))

    return result
class Solution(object):
    def count1Bits(self, num):
        result = 0
        while num:
            result += 1
            num = num & (num-1)

        return result

    def slowCount(self, num):
        result = 0
        while num:
            result += num % 2
            num = num >> 1
        return result
import numpy as np

# find the number of continuous subseq that sum to K

class Solution(object):
    def numSubseq(self, lst, K):
        countDict = {0: 1}
        numSeq = 0
        sumvals = 0
        for num in lst:
            sumvals += num
            remaining = K - sumvals
            numSeq += countDict.get(remaining, 0)
            countDict[sumvals] = 1 + countDict.get(sumvals, 0)

        return numSeq
# return True if it is possible to partition an array into N subseq
# each adding to K

class Solution(object):
    def recursiveFind(self, nums, used, parti, npartition, si, sum_val):
        if parti == nparition:
            return True
        if si == sum_val:
            return self.recursiveFind(nums, used, parti+1, npartition, 0, sum_val)
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            possible = self.recursiveFind(nums, used, parti, npartition, si+nums[i], sum_val)
            if possible:
                return True

            used[i] = False
        return False

    def partitionSum(nums, sum_val):
        nparition = np.sum(nums) // sum_val
        if np.sum(nums) % sum_val:
            return False

        used = np.zeros(len(nums), dtype=np.bool)
        return self.recursiveFind(nums, used, 0, nparition, 0, sum_val)


import numpy as np
import heapq

# Given a list [[num people, begin, end], [...]] find if it is possible to pick and drop off all passengers with total people in car not exceetin C

class Solution(object):
    def pickup(self, travelList, capacity):
        sorted(travelList, key=lambda item: item[1])

        numPass = 0
        ending = [] # contains tuples (end, num persons)
        for item in travelList:
            while ending and (ending[0] <= item[1]):
                end, nper = heapq.heappop(ending)
                numPass -= nper

            numPass += item[0]
            if numPass > capacity:
                return False
            heapq.heappush(ending, (item[2], item[0]))
        return True

    def fasterPickup(self, travelList, capacity):
        sz = np.max([item[2] for item in travelList])
        arr = np.zeros(sz, dtype=np.int32)
        for item in travelList:
            arr[item[1]] += item[0]
            arr[item[2]] -= item[0]

        arr = np.cumsum(arr)
        return np.all(arr <= capacity)
import numpy as np

# Given an array of integers, find if there exists indices i < j < k such
# that nums[i] < nums[j] > nums[k]

class Solution(object):
    def pattern132(self, nums: List[int]) -> bool:
        N = len(nums)
        if N < 3:
            return False

        stack = [(nums[0], None)] # monotone decreasing stack
        minIndex = 0
        for i in range(1, N):
            while len(stack) and (stack[-1][0] <= nums[i]):
                stack.pop()

            if len(stack) and (stack[-1][1] is not None):
                return True

            if nums[i] > nums[minIndex]:
                stack.append((nums[i], minIndex))
            else:
                stack.append((nums[i], None))
                if nums[i] < nums[minIndex]:
                    minIndex = i

        return False



import numpy as np

class Solution(object):
    # pacific ocean along W and S atlantic along E and N. grid of heights, find all squared where water can flow to both atlantic and pacific, water can only flow on non increasing heights E, W, N or S
    def dfs(self, heights, visited, stack):
        nrow, ncol = len(heights), len(heights[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while len(stack):
            r, c = stack.pop()
            visited.add((r, c))
            for direc in directions:
                row, col = r + direc[0], c + direc[1]
                if (row < 0) or (row == nrow) or (col < 0) or (col == ncol) or ((row, col) in visited) or (heights[row][col] < heights[r][c]):
                    continue
                stack.append((row, col))


    def flowSquares(self, heights: List[List[int]]) -> List[Tuple[int]]:
        visitedPac = set()
        stack = []

        for i in range(nrow):
            stack.append((i, 0))
        for j in range(1, ncol):
            stack.append((nrow-1, j))
        self.dfs(heights, visitedPac, stack)

        visitedAtl = set()
        for i in range(nrow):
            stack.append((i, ncol-1))
        for j in range(ncol-1):
            stack.append((0, j))

        self.dfs(heights, visitedAtl, stack)
        return list(visitedPac.intersection(visitedAtl))
        
       

# given a list of non-unique integers, return al unique permutations


class UniquePerm(object):
    def recursiveFind(self, list_len, result, counter, curr_res):
        if len(curr_res) >= list_len:
            result.append(curr_res.copy())
            return

        for word, freq in counter.items():
            # include word at position i0 in curr_res
            if freq == 0:
                continue
            counter[word] -= 1
            curr_res.append(word)

            self.recursiveFind(list_len, result, counter, curr_res)
            counter[word] += 1
            curr_res.pop()
        return


    def findAll(self, num_list):
        result = []
        counter = {}
        for num in num_list:
            counter[num] = counter.get(num, 0) + 1
        self.recursiveFind(len(num_list), result, counter, [])
        return result
import numpy as np

# Can +, - be placed in a number to sum to a value?

class SumToValue(object):
    def getPartialVals(self, n):
        digits = []
        powers = []
        mult = 1
        while n:
            rem = n%10
            digits.append(rem)
            powers.append(mult)
            mult *= 10
            n = n/10
        digits.reverse()
        powers.reverse()
        return np.array(digits, dtype=np.int), np.array(powers, dtype=np.int)

    def getValue(self, index, s):
        if self.nTable[index, s] != -1:
            return self.nTable[index, s]

        val = False
        dig_value = np.sum(np.multiply(self.digitArr[0:index+1], self.powArr[0:index+1]))
        dig_value = dig_value/self.powArr[index]

        val = False
        if index == 0:
            if (s == self.digitArr[0]) or (s == -self.digitArr[0]):
                val = True
        elif (s == dig_value) or (s == -dig_value):
            val = True
        elif (s > dig_value) or (s < -dig_value):
            val = False
        else:
            for j in range(index-1, -1, -1):
                dig_j = np.sum(np.multiply(self.digitArr[j:index+1], self.powArr[j:index+1]))
                dig_j = dig_j / self.powArr[index]
                val = val | self.getValue(j, s-dig_j) | self.getValue(j, s+dig_j)

        self.nTable[index, s] = 1 if val else 0
        return val


    def find(self, n, k):
        # N(i,s) = true if +/- can be placed in top i digits to sum to s
        #        = N(i-1, s-a[i]) | N(i-1, s+a[i])
        #          | N(i-2, s-a[i-1,i]) | N(i-1, s+a[i-1,i])
        #          | N(i-3, s-a[i-2,i-1,i]) | ...
        self.digitArr, self.powArr = self.getPartialVals(n)
        ndigits = self.digitArr.shape[0]

        self.nTable = np.zeros((ndigits, 2*(k+n)+1), dtype=np.int8)
                
        # 0: False, 1: True, -1: not set
        self.nTable[:, :] = -1
        return self.getValue(ndigits-1, k)

# serialize a binary tree using preorder trav

# Node_val(left_node)(right_node)

# If right node is none: node_val(left_node)
# if left node is none but not right node: node_val()(right_node)

class Preorder(object):
    def recStr(self, node, res):
        if res is None:
            return
        res.append(node.val)
        if (node.left is None) and (node.right is None):
            return
        if node.right is not None:
            res.append("(")
            self.recStr(node.left)
            res.append(")")
        res.append("(")
        self.recStr(node.right)
        res.append(")")
        

    def toStr(self, node):
        res = []
        self.recStr(node, res)
        return "".join(res)


import numpy as np
import heapdict # indexed priority queue

class Graph(object):
    def __init__(self, nvert):
        self.nVert = nvert
        self.edges = []
        for i in range(nvert):
            self.edges.append({})

    def addEdge(self, u, v, weight):
        self.edges[u][v] = weight
        self.edges[v][u] = weight

class MST(object):
    def primAlgo(self, graph):
        mst_edges = []
        pqueue = heapdict.heapdict()
        visited = np.zeros(graph.nVert, dtype=bool)
        u = 0
        visited[u] = True
        for v, weight graph.edges[u].items():
            if visited[v]:
                continue
            pqueue[(u,v)] = weight

        while len(pqueue):
            (u,v) = pqueue.popitem()
            if visited[v]:
                continue

            mst_edges.append((u,v))
            visited[v] = True
            for v2, weight in graph.edges[v].items():
                if visited[v2]:
                    continue
                pqueue[(v,v2)] = weight

        if not all(visited):
            return []
        return mst_edges

''' Remove K digits from a string with N digits to get the lowest possible number '''

from typing import List

class Solution(object):
    def removeKDigits(self, digits: str, k: int) -> str:
        if (len(digits) == 0) or (k >= len(digits)):
            return ""

        stack = [digits[0]]
        i = 0
        for i in range(1, len(digits)):
            while len(stack) and (k > 0) and (stack[-1] > digits[i]):
                stack.pop()
                k -= 1

            stack.append(digits[i])

        return "".join(stack[0:len(stack)-k])

# given a grid of 0 water and X land, remove all X surrounded by 0

class RemoveSurrounded(object):
    def dfs(self, board, row, col, visited):
        nrow, ncol = len(board), len(board[0])
        if (row < 0) or (row >= nrow) or (col < 0) or (col >= ncol) or ((row, col) in visited) or (board[row][col] != 'X'):
            return
        dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        visited.add((row, col))
        for dr in dirs:
            r, c = row + dr[0], col + dr[1]
            self.dfs(board, r, c, visited)



    def remove(self, board):
        nrow, ncol = len(board), len(board[0])
        visited = set()
        for i in range(nrow):
            if board[i][0] == 'X':
                self.dfs(board, i, 0, visited)
            if board[i][ncol-1] == 'X':
                self.dfs(board, i, ncol-1, visited)

        for j in range(1, ncol-1):
            if board[0][j] == 'X':
                self.dfs(board, 0, j, visited)
            if board[nrow-1][j] == 'X':
                self.dfs(board, nrow-1, j, visited)

        for i in range(nrow):
            for j in range(ncol):
                if (board[i][j] == 'X') and ((i, j) not in visited):
                    board[i][j] = '0'


# given a robot positioned at origin. find if a given sequence of steps
# when repeated multiple times can give a cycle

# G -> move forward, L -> turn left, R -> turn right

class RobotInCircle(object):
    def isCycle(self, instr):
        direc = [0, 1, 2, 3] # N, W, S, E
        loc = (0, 0)
        curr_dir = 0
        for step in instr:
            if step == "G":
                loc = self.stepAhead(loc, curr_dir)
            elif step == "L":
                curr_dir = self.turnLeft(curr_dir)
            elif step == "R":
                curr_dir = self.turnRight(curr_dir)
            else:
                raise ValueError("Invalid step")

        if curr_dir != 0:
            return True
        elif loc == (0, 0):
            return True
        return False


# convert to roman
class Roman(object):
    def convert(self, nums):
        roman_dict = {'I': 1, 'M': 1000}
        sum_val = 0
        mapped_vals = [roman_dict[va] for val in nums]
        for i in range(len(nums)-1):
            if mapped_vals[i] >= mapped_vals[i+1]:
                sum_val += mapped_vals[i]
            else:
                sum_val -= mappped_vals[i]

        sum_val += mapped_vals[len(nums)-1]
        return sum_val

#[[1,2],[3,4]] -> [[4, 1], [2, 3]]

class Solution(object):
    @staticmethod
    def to1DIndex(row, col, ncol):
        return row*ncol + col

    @staticmethod
    def to2DIndex(indx, ncol):
        i, j = indx // ncol, indx % ncol
        return i, j

    def rotate(self, twodarr):
        nrow, ncol = len(twodarr), len(twodarr[0])
        res = [[0] * ncol for i in range(nrow)]

        for i in range(nrow):
            for j in range(ncol):
                indx1D = Solution.to1DIndex(i, j, ncol)
                indx2D = Solution.to2DIndex((idx1D + 1) % (nrow*ncol))
                res[indx2D[0]][indx2D[1]] = twodarr[i][j]
        return res


# a 2d grid has oranges and rotten oranges and empty cells. Each second, all neighboring oranges around a rotten orange rot. After t seconds, get the number of fresh oranges

class Rotten(object):
    def count(board, N):
        nrow, ncol = len(board), len(board[0])
        queue = collections.deque()
        visited = {}
        for i in range(nrow):
            for j in range(ncol):
                if board[i][j] == 'r':
                    queue.append((i,j))
                    visited.add((i,j))

        dirs = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        for i in range(N):
            nchild = len(queue)
            for j in range(nchild):
                node = queue.popleft()
                if node in visited:
                    continue
                for direc in dirs:
                    ni, nj = node[0] + direc[0], node[1] + direc[1]
                    if (ni < 0) or (ni >= nrow) or (nj < 0) or (nj >= ncol) or ((ni, nj) in visited) or (board[ni][nj] != 'f'):
                        continue
                    visited.add((ni, nj))
                    queue.add((ni, nj))
                    board[ni][nj] = 'r'

        return sum([board[i][j] == 'f' for i in range(nrow) for j in range(ncol)])
# given a list of nodes with a list of outgoing edges. Find nodes that 
# can only lead to dead ends, i.e. nodes with no outlet

class SafeNodes(object):
    def isSafe(self, nodei, adj_list, safe_dict):
        if nodei in safe_dict:
            return safe_dict[nodei]

        safe_dict[nodei] = False
        for nbr in adj_list[nodei]:
            if not self.isSafe(nbr, adj_list, safe_dict):
                return False
        safe_dict[nodei] = True
        return True

    def find(self, adj_list):
        result = []
        safe_dict = {}

        for i in range(len(adj_list)):
            if self.isSafe(i, adj_list, safe_dict):
                result.append(i)

        return result
# given a list of jobs by characters A-Z, all jobs take equal time 1 unit to run, there is a cool down time between same type of jobs, find minimum time to execute all

from collections import Counter, deque
import heapq

class Solution(object):
    def minTime(self, jobs_list, interval):
        freq = Counter(jobs_list)

        freq = {}
        for jb in jobs_list:
            freq[jb] = 1 + freq.get(jb, 0)

        heap = [-f for f in freq.values()] # -freq
        queue = deque()  # (-freq, time)
        heap = heapq.heapify(heap)

        time = 0
        while heap or queue:
            time += 1
            most_freq = heapq.heappop(heap)
            most_freq += 1 # perform the job
            if most_freq == 0:
                continue
            queue.append((most_freq, time + interval))

            if queue[0][1] == time:
                f, tm = queue.popleft()
                heapq.heappush(heap, f)

        return time

    # Solve if different jobs have different execute times
    def minTime(self, jobs, cool_off, queue, time, min_time, memo={}):
        if (not jobs) and (not queue):
            return min_time

        key = self.getKey(jobs, queue, time)
        if key in memo:
            return memo[key]

        if len(queue) and (queue[0][1] == time):
            job, tm = queue.popleft()
            jobs.append(job)

        jobs_in_cooloff = set([j[0] for j in queue])
        
        # wait for 1 unit
        wait_time = self.minTime(jobs, queue, time+1, min_time+1)

        # process one job
        tm = float("inf")
        for ind, jb in enumerate(jobs):
            jb_name, jb_time, jb_cooloff = jb
            queue.append((jb_name, jb_cooloff))
            tm = min(tm, self.minTime(jobs[0:ind] + jobs[ind+1:], queue, time+1, min_time+jb_time))

        memo[key] = min(wait_time, tm)
        return memo[key]




import numpy as np

# given a list of words and a word, return a max of 3 search suggestions per character of the word

class Solution(object):
    def sugestions(self, wordlist, word):
        wordlist = sorted(wordlist)
        result = []
        left, right = 0, len(wordlist) - 1
        for i in range(len(word)):
            while ((left < right) and (len(wordlist[left]) > i) and (wordlist[left][i] != word[i])):
                left += 1

            while ((left < right) and (len(wordlist[right]) > i) and (wordlist[right][i] != word[i])):
                right -= 1

            numwords = min(3, right - left + 1)
            result.append(wordlist[left:right+1])

        return result


# complete 1 task in 1 min
# complete n/2 tasks in 1 min if n % 2 == 0
# complete 2n/3 tasks in 1 minute if n % 3 == 0

class ShortestTime(object):
    def find(self, tasks):
        time_dict = {0: 0, 1: 1}
        return self.recursiveFind(tasks, time_dict)

    def recursiveFind(self, tasks, time_dict):
        if tasks in time_dict:
            return tie_dict[tasks]

        red1 = 1 + self.recursiveFind(tasks-1, time_dict)
        red2 = tasks
        if tasks % 3 == 0:
            red2 = 1 + self.recursiveFind(tasks // 3, time_dict)
        elif tasks % 2 == 0:
            red2 = 1 + self.recursivefind(tasks // 2, time_dict)
        result = min(red1, red2)
        time_dict[tasks] = result
        return result
# find the length of the shortest window in a string containing all the characters in a word

class ShortestWindow(object):
    def shortestWinLen(self, text, find_word):
        min_len = len(text) + 1
        left, right = 0, 0
        ch_count = {}
        for ch in find_word:
            ch_count[ch] = ch_count.get(ch, 0) + 1

        need_chars = len(ch_count.keys)
        have_chars = 0
        text_count = {}
        while right < len(text):
            ch = text[right]
            if ch not in ch_count:
                right += 1
                continue
            text_count[ch] = text_count.get(ch, 0) + 1
            if text_count[ch] == ch_count[ch]:
                have_chars += 1

            while (have_chars == need_chars) and (left <= right):
                min_len = min(min_len, right - left + 1)
                ch_left = word[left]
                if ch_left not in ch_count:
                    left += 1
                    continue
                if text_count[ch_left] == ch_count[ch_left]:
                    have_chars -= 1
                text_count[ch_left] -= 1
                left += 1

            right += 1
        return min_len






def slowsum(arr):
    aug_arr = sorted(arr, key=lambda x: -x)
    return np.cumsum(aug_arr)

# find the least number of steps to reach board end when you move using a die (6 face) throw, with snakes and ladders on the board

class SnakedAndLadders(object):
    def minMoves(self, board):
        pos = 1
        N = len(board)
        final_pos = N * N
        visited = {}
        queue = deque()
        queue.append((pos, 0)) # position, current number of moves
        while queue:
            cpos, moves = queue.popleft()
            if cpos in visited:
                continue
            if cpos == final_pos:
                return moves
            for i in range(1, 7):
                new_pos = cpos + i
                r, c = self.getRowCol(new_pos, N)
                while board[r][c] != -1:
                    new_pos += board[r][c]
                    r, c = self.getRowCol(new_pos, N)
                queue.append((new_pos, moves+1))
        return None

    def getRowCol(self, pos, N):
        if pos % 2 == 0:
            return pos // N, pos % N
        return pos // N, N - pos % N - 1
import numpy as np

# a sorted (ascending) array is rotated k times to the right. Find the minimum element

class Solution(object):
    def minInSortRotArr(self, arr):
        left, right = 0, len(arr) -1
        while left <= right:
            pivot = (left + right) // 2
            if arr[left] <= arr[right]:
                return arr[left]
            if arr[pivot] > arr[left]:
                left = pivot + 1
            else:
                right = pivot - 1

        
        
# given a 2D matrix, visit in spiral order

class Solution(object):
    def spiralVisit(self, matrix):
        result = []
        if len(matrix) == 0:
            return result
        rowBounds = [0, len(matrix)]
        colBounds = [0, len(matrix[0])]

        while ((rowBounds[1] > rowBounds[0]) and (colBounds[1] > colBounds[0])):
            for j in range(colBounds[0], colBounds[1]):
                result.append(matrix[rowBounds[1]-1][j])
            rowBounds[1] -= 1

            for i in range(rowBounds[1], rowBounds[0]-1, -1):
                result.append(matrix[i][colBounds[1]-1])
            colBounds[1] -= 1

            for j in range(colBounds[1], colBounds[0]-1, -1):
                result.append(matrix[rowBounds[0]][j])
            rowBounds[0] += 1

            for i in range(rowBounds[0], rowBounds[1]):
                result.append(matrix[i][colBounds[0]])
            colBounds[0] += 1

        return result


# given a word and a list of words, find the minimum words whose letters can
# be used to recreate the word

import numpy as np

class Solution(object):
    # M(word, i) : minimum words from i to N-1 that can be used to create
    # word
    # M(word, i) = min(M[word, i+1], 1 + M[word - letters from word[i], i+1])

    def findValue(self, word, letters, fromwd, counters, mtable, N):
        if fromwd >= N:
            return 0
        positions = self.extractBits(letters)
        chars = word[positions]
        if self.canConstruct(chars, counters[fromwd]):
            mtable[letters, fromwd] = 1
        if mtable[letters, fromwd] != -1:
            return mtable[letters, fromwd]
        # use word fromwd
        newmask = self.getRemChars(chars, counters[fromwd])
        val = min(self.findValue(word, letters, fromwd+1, counters, mtable, N),
                1 + self.findValue(word, newmask, fromwd+1, counters, mtable, N))
        mtable[letters, fromwd] = val
        return val


    def minWords(self, word, wordlist) -> int:
        wlen = len(word)
        N = len(wordlist)

        counters = []
        for wd in wordlist:
            counters.append(collections.Counter(wd))

        mask = 1 << wlen
        mtable = np.full((mask, N), -1, dtype=np.int32)
        mtable[0, :] = 0

        return self.findValue(word, mask-1, 0, counters, mtable, N)
import numpy as np

class Graph(object):
    def __init__(self, num_vtx):
        self.graph = [None] * num_vtx

    def addEdge(self, from_vt, to_vt):
        if self.graph[from_vt] is None:
            self.graph[from_vt] = []

        self.graph[from_vt].append(to_vt)


class StrongConnComp(object):
    def topSortDFS(self, nd_num, graph, sort_vtx, pos):
        if visited[nd_num]:
            return pos
        visited[nd_num] = True
        for nbr in graph.graph[nd_num]:
            pos = self.topSortDFS(nbr, graph, sort_vtx, pos)
        sort_vtx[pos] = nd_num
        return pos-1


    def topSort(self, graph):
        nvtx = len(graph.graph)
        visited = np.zeros(nvtx, dtype=bool)
        sort_vtx = np.zeros(nvtx, dtype=np.int)
        next_id = 0
        pos = nvtx - 1
        for i in range(nvtx):
            pos = self.topSortDFS(i, graph, sort_vtx, visited, pos)
        return sort_vtx


    def dfs(self, vt, stack, on_stack, graph, visited, low_link):
        if visited[vt]:
            return
        visited[vt] = True
        stack.append(vt)
        on_stack[vt] = 1
        nbrs = graph.graph[vt]
        for nbr in nbrs:
            self.dfs(nbr, stack, on_stack, graph, visited, low_link)
            if (low_link[nbr] <= low_link[vt]) and (on_stack.get(vt, 0) == 1):
                low_link[vt] = low_link[nbr]

        if low_link[vt] == vt:
            nd = -1
            while nd != vt:
                nd = stack.pop()
                on_stack[nd] = 0


    
    def findSCC(self, graph):
        sort_vtx = self.topSort(graph)
        nvtx = len(graph.graph)
        visited = np.zeros(nvtx, dtype=bool)
        id_arr = np.zeros(range(nvtx), dtype=np.int)
        for i in range(nvtx):
            id_arr[sort_vtx[i]] = i
            
        low_link = id_arr #np.array(range(nvtx), dtype=np.int)
        stack = []
        results = []
        on_stack = {}
        idnum = 0
        # assign ids first
        for i in range(nvtx):
            if visited[i]:
                continue
            self.dfs(i, stack, on_stack, graph, visited, low_link)

        components = {}
        for i in range(nvtx):
            comp = low_link[i]
            if comp not in components:
                components[comp] = []
            components[comp].append(i)

        for k,v in components.items():
            if len(v) > 1:
                results.append(v)

        return results

# return if a subseq of length > 1 exists in an array of positive numbers if the sum of subseq is a multiple of K

class SubseqMultiple(object):
    def find(self, nums, K):
        rem_dict = {}
        for i, num in enumerate(nums):
            rem = num % K
            if rem in rem_dict and i - rem_dict[rem] > 1:
                return True
            elif rem not in rem_dict:
                rem_dict[rem] = i
        return False
'''Write code to implement a tree structure that supports locking of nodes.
Tree is defined by a list of parent nodes for each node. Each node has one parent except for root.
Lock a node by a user. A node can be locked if it is unlocked. Also none of its parents must be locked and at least one of its descendants must be locked
Unlock a node with a user
'''

class TreeNode(object):
    def init(self, val):
        self.val = val


class structure(object):
    def __init__(self, parents: List[int]):
        self.nnode = len(parents)
        self.parents = parents
        self.children = defaultdict(list)
        for I, pr in enumerate(parents):
            If pr == -1:
                continue

            self.children[pr].append(I)
        self.locked = [None] * self.nnode

    def lock(self, node, user):
        if self.locked[node] is not None:
            return False

        pr = self.parents[node]
        while pr != -1:
            if self.locked[pr] is not None:
                return False

        nlocked = 0
        queue = deque()
        queue.append(node)
        while queue:
            for child in self.children[node]:
                if self.locked[child]:
                    self.locked[child] = None
                    nlocked += 1
                queue.append(child)

        if nlocked == 0:
            return False

        self.locked[node] = user
        return True

#Find the number of binary search trees that can be constructed using numbers 1, 2, n

class NBT(object):
    def numTrees(self, ntree):

        numTable = np.ones(ntree+1, dtype=np.int32)
        for num in range(ntree+1):
            count = 0
            for root in range(1, num+1):
                count += numTable[root-1] * numTable[num-root]
            numTable[root] = count
        return numTable[ntree]


#Given a list of triplets, return true if a target triplet can be constructed from the inputs. Only operation allowed is taking a max of each element of a triplet to create a new triplet from a pair of triplets.

class Possible(object):
    def find(self, tripletList, targetTriplet):
        indices = [None, None, None]
        for i, trip in enumerate(tripletList):
            for j in range(3):
                if trip[j] == targetTriplet[j]:
                    valid = True
                    for k in range(3):
                        if j == k:
                            continue
                        if trip[k] > targetTriplet[k]:
                            valid = False
                    if valid:
                        indices[j] = i

        return all([t is not None for t in indices])


#In a string comprising of (, ), and * where * can be replaced with (, ) or _, find if the sequence is a valid string or not.
#Table stores the number of closing brackets to the right of pos assuming POs has ( or )

class Valid(object):
    def find(self, wstr):
        minleft, maxleft = 0, 0
        for s in wstr:
            if s == '(':
                minleft += 1
                maxleft += 1
            elif s == ')':
                minleft -= 1
                maxleft -= 1
            else:
                minleft -= 1
                maxleft += 1
            if maxleft < 0:
                return False
            if minleft < 0:
                minleft = 0
        return True


#find if a board is valid soduku
# numbers 1 - 9

def valid(board):
    cols = np.zeros((9, 9), dtype=np.int32) # or use collections.defaultdict(set) 
    rows = np.zeros((9, 9), dtype=np.int32)
    subsq = np.zeros((3, 3, 9), dtype=np.int32)

    nrow, ncol = len(board), len(board[0])
    for i in nrow:
        for j in ncol:
            if board[i][j] == '.':
                continue
            indx = int(board[i][j])
            if cols[j, indx-1] != 0:
                return False
            if rows[i, indx-1] != 0:
                return False
            if subsq[i // 3, j // 3, indx-1] != 0:
                return False
            cols[j, indx-1] = indx
            rows[i, indx-1] = indx
            subsq[i//3, j//3, indx-1] = indx
    return True
import os
'''
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree [3,9,20,null,null,15,7],
'''

class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        assert(isinstance(left, TreeNode))
        self.left = left
        self.right = right

class Solution(object):
    def levelOrder(self, root):
        result = []
        i = 0
        queue = [root, 1]
        level = []
        while i != len(queue):
            node = queue[i]
            i += 1
            if (node == 1) and (i != len(queue)):
                queue.append(1)
                result.append(level)
                level = []
            else:
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        if level:
            result.append(level)

        return result
# given a list of triplets and a target triplet. Allowed operations are taking the maximum of each number of a triplet

class Soln(object):
    @staticmethod
    def possible(tripletList, targetTriplet):
        indices = [[], [], []]
        for i, triplet in enumerate(tripletList):
            for j in range(3):
                if triplet[j] == targetTriplet[j]:
                    indices[j].append(i)


        for i in range(3):
            if len(indices[i]) == 0:
                return False
            for j in range(3):
                if j == i:
                    continue
                if indices[i][0] > indices[j][0]:
                    return False
        return True


# Given an array of N tuples, each with 2 elements: cost of going to city A and
# city B for a candidate, find the minimum cost of flying half candidates to 
# city A and other half to city B. Assume N is even

from typing import List

# Can also do DP

class Solution(object):
    def getDPVal(self, a, b, table, N, costs):
        if a < 0:
            return 0
        if b < 0:
            return 0
        if table[a, b] != -1:
            return table[a, b]
        i = N - a - b
        val = min(costs[i][0] + self.getDPVal(a-1, b, table, N, costs),
                costs[i][1] + self.getDPVal(a, b-1, table, N, costs))
        table[a, b] = val
        return val

    def minCostDP(self, costs: List[List[int]]) -> int:
        # T(i, a, b) = min(c[i][0] + T(i+1, a-1, b), c[i][1] + T(i+1, a, b-1))
        # i = N - a - b
        n = len(costs)
        table = np.full((n//2, n//2), -1, dtype=np.int32)
        return self.getDPVal(n//2, n//2, table, n, costs)

    def minCost(self, costs: List[List[int]]) -> int:
        diffCost = []
        for i, abcosts in enumerate(costs):
            acost, bcost = abcosts
            diffCost.append((bcost - acost, i))

        diffCost.sort(key = lambda x: x[0])
        totalCost = 0
        n = len(costs)
        for i in range(n/2):
            totalCost += costs[diffCost[i][1]][1]

        for i in range(n/2, n):
            totalCost += costs[diffCost[i][1]][0]

        return totalCost
# given a string of integers, return the number of valid IP addresses that can be generated by placing 3 dots, partitioning into 4 numbers between 0 and 255

class NumValidIPAddr(object):
    def recursiveFind(self, nums, dots, begin, result, curr_res):
        if dots == 0:
            if begin == len(nums):
                return
            val = int(nums[begin:])
            if val < 256:
                result.append(curr_res + nums[begin:])
            return

        for i in range(begin+1, len(nums)): # couls also do min(begin+3, len(nums))
            num = int(nums[begin:i])
            if num < 256:
                orig_len = len(curr_res)
                curr_res.append(nums[begin:] + ".")
                self.recursiveFind(nums, dots-1, i+1, result, curr_res)
                curr_res = curr_res[0:orig_len]
            else:
                break


    def validIPs(self, nums):
        result = []
        curr_res = ""
        self.recursiveFind(nums, 3, 0, result, curr_res)
        return result
# find the disting sums that can be formed using ints from list any number of times to sum to a number

class Solution(object):
    def dfs(self, int_list, val, begin, memo):
        if (val < 0) or (begin >= len(int_list)):
            return None
        if val == 0:
            return []
        if (val, begin) in memo:
            return memo[(val, begin)]

        result = []
        for i in range(begin, len(int_list)):
            rem_sum = val - int_list[i]
            res = self.dfs(int_list, rem_sum, i, memo, result)
            if res is not None:
                for r in res:
                    r.append(int_list[i])
            result.extend(res)
        memo[(val, begin)] = result
        return result

    def distinctSums(self, int_list, val):
        memo = {} # sum, index in int_list (begin)

        result = []
        return self.dfs(int_list, val, 0, memo, result)
import io
import itertools
import numpy as np
import os
import re
import string
import tensorflow as tf
import tqdm

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorizationSEED = 42 
AUTOTUNE = tf.data.AUTOTUNE

sentence = "The wide road shimmered in the hot sun"
tokens = list(sentence.lower().split())
print(len(tokens))

vocab, index = {}, 1 # start indexing from 1
vocab['<pad>'] = 0 # add a padding token 
for token in tokens:
  if token not in vocab: 
    vocab[token] = index
    index += 1
vocab_size = len(vocab)
print(vocab)

inverse_vocab = {index: token for token, index in vocab.items()}
print(inverse_vocab)

example_sequence = [vocab[word] for word in tokens]

window_size = 2
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
      example_sequence, 
      vocabulary_size=vocab_size,
      window_size=window_size,
      negative_samples=0)
print(len(positive_skip_grams))

for target, context in positive_skip_grams[:5]:
  print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")

target_word, context_word = positive_skip_grams[0]

# Set the number of negative samples per positive context. 
num_ns = 4

context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))
negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
    true_classes=context_class, # class that should be sampled as 'positive'
    num_true=1, # each positive skip-gram has 1 positive context class
    num_sampled=num_ns, # number of negative context words to sample
    unique=True, # all the negative samples should be unique
    range_max=vocab_size, # pick index of the samples from [0, vocab_size]
    seed=SEED, # seed for reproducibility
    name="negative_sampling" # name of this operation
)
print(negative_sampling_candidates)
print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])

# Add a dimension so you can use concatenation (on the next step).
negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

# Concat positive context word with negative sampled words.
context = tf.concat([context_class, negative_sampling_candidates], 0)

# Label first context word as 1 (positive) followed by num_ns 0s (negative).
label = tf.constant([1] + [0]*num_ns, dtype="int64") 

# Reshape target to shape (1,) and context and label to (num_ns+1,).
target = tf.squeeze(target_word)
context = tf.squeeze(context)
label =  tf.squeeze(label)

print(f"target_word     : {inverse_vocab[target_word]}")
print(f"context_indices : {context}")
print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
print(f"label           : {label}")

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence, 
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)
    
    # Iterate over each positive skip-gram pair to produce training examples 
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1, 
          num_sampled=num_ns, 
          unique=True, 
          range_max=vocab_size, 
          seed=SEED, 
          name="negative_sampling")
      
      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

with open(fl) as f:
      lines = f.read().splitlines()
for line in lines[:20]:
  print(line)

# We create a custom standardization function to lowercase the text and 
# remove punctuation.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')

# Define the vocabulary size and number of words in a sequence.
vocab_size = 4096
sequence_length = 10

text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Set output_sequence_length length to pad all samples to same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

class Word2Vec(Model):
    def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = Embedding(vocab_size, 
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding", )
    self.context_embedding = Embedding(vocab_size, 
                                       embedding_dim, 
                                       input_length=num_ns+1)
    self.dots = Dot(axes=(3,2))
    self.flatten = Flatten()

  def call(self, pair):
    target, context = pair
    we = self.target_embedding(target)
    ce = self.context_embedding(context)
    dots = self.dots([ce, we])
    return self.flatten(dots)

embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
import numpy as np

# Given a list of dictionary words, find if it is possible to break a word
# into subwords that are present in dictionary
# a work from dictionary can be reused

class Solution(object):
    def constructFromIndex(self, word, wordSet, begin, possible):
        if begin >= len(word):
            return False
        if possible[begin] != -1:
            return possible[begin]

        val = False
        for wd in wordSet:
            if (begin + len(wd) > len(word)) or (word[begin:begin+len(wd)] != wd):
                continue
            val = self.constructFromIndex(word, wordSet, begin+len(wd), possible)
            if val:
                break
        possible[begin] = val
        return val

    def wordBreak(self, word, wordList):
        wordSet = set(wordList)
        # possible[i] = True if word[i:] can be constructed using wordList
        possible = np.zeros(len(word), dtype=np.int8)
        possible[;] = -1 #uninitialized
        return self.constructFromIndex(word, wordSet, 0, possible)


'''
Given two words (beginWord and endWord), and a dictionary's word list, find all shortest transformation sequence(s) from beginWord to endWord, such that:

Only one letter can be changed at a time
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
Note:

Return an empty list if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
Example 1:

Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output:
[
  ["hit","hot","dot","dog","cog"],
  ["hit","hot","lot","log","cog"]
]
Example 2:

Input:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

Output: []

Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.
'''

from collection import deque

class GraphNode(object):
    def __init__(self, word):
        self.word = word
        self.parent = None


class Solution(object):
    def appendResult(self, result, node):
        res = []
        while node is not None:
            res.append(node.word)
            node = node.parent
        result.append(res.reverse())

    def countDiff1(self, word1, word2):
        diffCnt = 0
        for ch1, ch2 in zip(word1, word2):
            if ch1 != ch2:
                diffCnt += 1
                if diffCnt > 1:
                    return False
        return diffCnt == 1

    def solve(self, beginWord, endWord, wordList):
        wordQueue = deque()
        GraphNode.NCHILDREN = len(wordList)
        tree = Node(beginWord)
        wordQueue.append((beginWord, tree))
        result = []
        visited = {beginWord}
        if endWord not in wordList:
            return None
        if endWord == beginWord:
            return result
        
        resultCount = 0
        tree = None
        while len(wordQueue):
            nodesCount = len(wordQueue)
            for i in range(nodesCount):
                word, parNode = wordQueue.popleft()
                
                if word == endWord:
                    resultCount += 1
                    self.appendResult(result, parNode)
                else:
                    for lword in wordList:
                        if (lword not in visited) and (self.countDiff1(word, lword)):
                            nd = GraphNode(lword)
                            nd.parent = parNode
                            wordQueue.append((lword, nd))
                            visited.add(lword)
            if resultCount != 0:
                break

        return result

# given a begin word and a destination word, find the shortest transition using one letter transitions, such that all intermediate words are present in a list

from typing import List
feom collections import deque, defaultdict

class Solution(object):
    def createAdjDict(self, wset):
        adjDict = defaultdict(list)
        for word in wset:
            for j in range(len(word)):
                wnode = word[0:j] + '_' + word[j+1:]
                adjDict[wnode].append(word)
        return adjDict


    def shortestTrans(self, source: str, destin: str, words: List[str]) -> int:
        wset = set(words)
        if destin not in wset:
            return -1

        adjDict = self.createAdjDict(wset)

        queue = deque([source])
        visited = set([source])
        nchange = 0

        while queue:
            nchild = len(queue)
            for i in range(nchild):
                word = queue.popleft()
                if word == destin:
                    return nchange
                for j in range(len(word)):
                    wnode = word[0:j] + '_' + word[j+1:]
                    if wnode not in adjDict:
                        continue
                    for child in adjDict[wnode]:
                        if child not in visited:
                            visited.add(child)
                            queue.append(child)
            nchange += 1

        return -1

Question: Given an array, find if there is a sub-array with 0 sum.

# P[i, S] = True if A[0, 1, ... i] has a subarray with sum S, False otherwise
# P[i, S] = P[i-1, S] (do not include A[i]) | P[i-1, S-A[i]] (include A[i])
# final answer P[N-1, 0]

import numpy as np

class SumZero(object):
    def __init__(self, A):
        self.A = A
        self.S = np.sum(np.abs(self.A))
        
    def findTableValue(self, i, s, table):
        if i < 0:
            return 0
        if table[i, s + self.S-1] != -1:
            return table[i, s + self.S - 1]
            
        table[i, s + self.S - 1] = self.findTableValue(i-1, s)
        if table[i, s + self.S - 1] == 0:
            table[i, s + self.S - 1] = self.findTableValue(i-1, s-self.A[i])
        return table[i, s + self.S - 1]
        
    def isSubArrZeroSum(self):
        table = np.zeros((len(self.A), 2*self.S), dtype=np.int8)
        table[:, :] = -1  # -1 unitialized, 0: False, 1: True
        return self.findTableValue(len(self.A)-1, 0, table)
        
        
        
        
# [-1, -2, 3, 4, 6]

# table = 5 X 32 dimensional table of chars
# table[:, :] = -1
# line 17 table[4, 0+16-1] = getTableValue(3, 0)
# if table[4, 0+16-1] is 0 (i.e. false), then calculate getTableValue(3, 0-6)
# Space and time O(N x S)


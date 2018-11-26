import numpy as np


class AstarPlanner:
    """ This file implements a basic A* path planner
     The function returns a planner 'object' that is initialized with the
     given obstacle map, start and goal position, but no path is planned at
     this stage.
     The actual planning is done in the 'member function' run(), which returns
     the waypoint sequence and optionally some debug information.
     """
    def __init__(self, map, start, goal):
        # some abbreviations
        self.map = map
        self.sz = map.shape
        self.w = self.sz[1]
        self.h = self.sz[0]

        # A* per-cell data
        self.closed = np.full(map.shape,
                              False)  # boolean matrix: true for each cell in closed list (initially, no cell is closed)
        self.g_cost = np.zeros(map.shape)  # matrix of g_cost. Only cells in open/closed list contain meaningful values.
        self.f_cost = np.zeros(map.shape)
        # matrix of f_cost = g_cost + heuristic. This is initialized and updated for each cell in the open list
        # Note: the heuristic (h_cost) is not explicitely stored, since it can always be retrieved from f_cost - g_cost

        self.parent = np.zeros(map.shape,
                               dtype=np.uint32)  # matrix of 'predecessor' cells (valid for open/closed cells only)
        # variables implementing the Open List as a Binary Heap
        self.openHeap = np.zeros(map.size, dtype=np.int32)  # vector of indices of all cells in the open list
        # The vector is always initialized to maximum size...

        # nOpenCells = np.uint32(0)  # but only the first nOpenCells are occupied
        self.nOpenCells = 0
        self.openHeapPos = np.zeros(map.shape, dtype=np.uint32)
        # matrix of positions of cells in openHeap. Only valid for open cells.
        # This 'back-reference' is required for updating the heap after after cost changes

        self.errorMessage = ''

        self.start = np.round(start)
        self.goal = np.round(goal)
        # start & goal as x/y coordinates and cell index
        self.x_start = self.start[0]
        self.y_start = self.start[1]
        self.startCell = np.ravel_multi_index([[self.y_start], [self.x_start]], self.sz)  # sub2ind
        # Converts a tuple of index arrays into an array of flat indices, applying boundary modes to the multi-index.

        self.x_goal = self.goal[0]
        self.y_goal = self.goal[1]
        self.goalCell = np.ravel_multi_index([[self.y_goal], [self.x_goal]], self.sz)

        if self.start.any() < 0 or self.start[0] > self.w or self.start[1] > self.h:
            self.errorMessage = 'Start position is not in map'
            # raise ValueError('Start position is not in map')
        elif self.goal.any() < 0 or self.goal[0] > self.w or self.goal[1] > self.h:
            self.errorMessage = 'Goal position is not in map'
            # raise ValueError('Goal position is not in map')
        elif map[self.start[1], self.start[0]]:
            self.errorMessage = 'Start position blocked'
            # raise ValueError('Start position blocked')
        elif map[self.goal[1], self.goal[0]]:
            self.errorMessage = 'Goal position blocked'
            # raise ValueError('Goal position blocked')
        else:
            # Initialize the open list (Heap) with the start cell
            self.g_cost.reshape(-1)[self.startCell] = 0
            self.delta = np.abs([self.x_start - self.x_goal, self.y_start - self.y_goal])  # 始末点距离
            self.f_cost.reshape(-1)[self.startCell] = 10 * np.abs(self.delta[0] - self.delta[1]) + 14 * np.min(
                self.delta)
            self.openHeap_add(self.startCell)

    # member function that does the actual planning
    # For better understanding, the number of iterations performed in each
    # invocation can be limited.
    #
    # return values:
    # waypoints: 2xN array the coordinates of the cells that constitute
    #            the path. One [x;y] pair per column. Empty if a path
    #            has not been found (yet)
    # errMsg, cellTypes, heapHead: debug data, see comments in code below
    #                              for code below for further information

    def run(self, *args):
        if not len(args):
            iterations = np.inf
        else:
            iterations = args[0]

        waypoints = []

        if not len(self.errorMessage):
            # Prepare some structures for efficient neighbourhood processing
            # Order: 6 5 4
            #        7 X 3
            #        0 1 2
            neighbourCosts = [14, 10, 14, 10, 14, 10, 14, 10]  # cost between cell and neighbour
            neighbourOffsets = [self.w - 1, self.w, self.w + 1, 1, -self.w + 1, -self.w, -self.w - 1, -1]
            # vector offset from cell to neighbour.

            # Run A* iterations, while the solution has not been found and there are still open cells
            while ~self.closed.reshape(-1)[self.goalCell][0] and self.nOpenCells > 0 and iterations > 0:
                iterations = iterations - 1
                # TODO: Remove open cell with least f_cost and put it into Closed List
                # 1.) Remove open cell with least f_cost and put it into Closed List
                # use self.openHeap_popHead(), self.closed.reshape(-1)
                cell = []  # TODO

                # 2.) Determine set of existing neighbours.
                [y, x] = np.unravel_index(cell, self.sz)
                validNeighbours = np.full(8, True)
                if x == 1:  # ‼️
                    validNeighbours[0][[0, 6, 7]] = False
                if x == self.w - 1:
                    validNeighbours[0][[2, 3, 4]] = False
                if y == 1:  # ‼️
                    validNeighbours[0][[0, 1, 2]] = False
                if y == self.h - 1:
                    validNeighbours[0][[4, 5, 6]] = False

                for i in range(8):
                    if validNeighbours[i]:
                        neighbourCell = cell + neighbourOffsets[i]

                        # Ignore obstacles and closed cells
                        if self.map.reshape(-1)[neighbourCell] or self.closed.reshape(-1)[neighbourCell]:
                            continue

                        heapPos = self.openHeapPos.reshape(-1)[neighbourCell]
                        if heapPos > 0:
                            # If a cell has a valid heapPos (> 0), it is in the Open List (Heap)
                            # --> check if its g_cost can be reduced
                            # TODO: check if its g_cost can be reduced
                            # use  self.g_cost.reshape(-1), self.f_cost.reshape(-1), self.parent.reshape(-1)
                            # self.openHeap_propagateToHead()

                            g_cost_reduction = []  # TODO

                        else:
                            # cell is yet unknown -> initialize it and add
                            # it to the Open Heap
                            # TODO: initialize cell and add it to the Open Heap
                            # use  self.g_cost.reshape(-1),   self.f_cost.reshape(-1), self.parent.reshape(-1)
                            # self.openHeap_add()
                            # np.unravel_index(), np.abs(), np.min()
                            self.g_cost.reshape(-1)[neighbourCell] = []  # TODO
                            [y, x] = np.unravel_index(neighbourCell, self.sz)
                            delta = []  # TODO
                            h_cost = []  # TODO
                            self.f_cost.reshape(-1)[neighbourCell] = []  # TODO
                            self.parent.reshape(-1)[neighbourCell] = []  # TODO
                            self.openHeap_add(neighbourCell)
                # end of A* loop

            if self.closed.reshape(-1)[self.goalCell][0]:
                # TODO: extract and return the path
                # we found a path - extract and return it
                idx = []  # TODO
                waypointCells =[]  # TODO
                # path length guess (exact size does not matter, preallocation prevents Lint warning)
                nCells = 0
                while idx != 0:
                    waypointCells[nCells] = []  # TODO
                    nCells = []  # TODO
                    idx = []  # TODO
                [Y, X] = np.unravel_index(waypointCells[np.arange(nCells - 1, -1, -1)].astype(np.int), self.sz)
                waypoints = np.vstack((X, Y))
            elif self.nOpenCells == 0:
                # no path found and no open cells left -> there is no
                # unblocked path between start and goal
                self.errorMessage = 'No Path found!'
                # raise ValueError('No Path found!')

        # prepare debug data, if requested by the caller
        errMsg = self.errorMessage

        # Map of cell types with values:
        # 0 = unknown (free) cell
        # 1 = blocked cell (obstacle)
        # 2 = cell in Open List
        # 3 = cell in Closed List
        # 4 = Open Cell with min. f_cost (head of Open Heap)
        cellTypes = np.uint8(self.map)
        cellTypes[self.openHeapPos > 0] = 2
        cellTypes[self.closed] = 3
        if self.nOpenCells > 0:
            cellTypes.reshape(-1)[self.openHeap[0]] = 4

        #  x/y coordinates of the Open Cell with min. f_cost
        if self.nOpenCells > 0:
            [y, x] = np.unravel_index(self.openHeap[0], self.sz)  # ind2sub
            heapHead = np.double(np.array([[x], [y]]))
        else:
            heapHead = np.zeros((2, 0))

        return waypoints, errMsg, cellTypes, heapHead

    # member function for accessing detailed information for the given cell
    def getCellInfo(self, x, y):  # This function is not used now
        cellInfo = {}
        cell = np.ravel_multi_index([[y], [x]], self.sz)
        cellInfo['g_cost'] = self.g_cost[cell]
        cellInfo['f_cost'] = self.f_cost[cell]
        if self.map[cell]:
            cellInfo['type'] = 'Obstacle'
        elif self.closed[cell]:
            cellInfo['type'] = 'Closed'
        elif self.openHeapPos[cell] > 0:
            cellInfo['type'] = 'Open'
        else:
            cellInfo['type'] = 'Unknown'
        parentCell = self.parent[cell]
        if parentCell > 0:
            [yParent, xParent] = np.unravel_index(parentCell, self.sz)
            cellInfo['parent'] = np.vstack((xParent, yParent))

        return cellInfo

    # ======= Functions for managing the Open List as a Binary Heap =======
    # A heap is a semi-ordered tree structure, where no child has a smaller
    # value than its parent

    # Add a cell to the Heap
    def openHeap_add(self, cell):
        # append new cell at the end...
        self.openHeap[self.nOpenCells] = cell
        self.nOpenCells = self.nOpenCells + 1
        self.openHeapPos.reshape(-1)[cell] = self.nOpenCells
        # ... and propagate it upwards
        self.openHeap_propagateToHead(cell)

    # Propagate cell upwards until the heap property
    # is restored (used by openHeap_add and after the f_cost reduction)
    def openHeap_propagateToHead(self, cell):
        idx = self.openHeapPos.reshape(-1)[cell]
        while idx > 1:

            idxParent = int(idx / 2)  # C: idx >> 1
            if self.f_cost.reshape(-1)[self.openHeap[idx - 1]] <= self.f_cost.reshape(-1)[self.openHeap[idxParent - 1]]:
                # swap with parent
                self.openHeapPos.reshape(-1)[self.openHeap[idx - 1]] = idxParent
                self.openHeapPos.reshape(-1)[self.openHeap[idxParent - 1]] = idx
                temp = self.openHeap[idx - 1]
                self.openHeap[idx - 1] = self.openHeap[idxParent - 1]
                self.openHeap[idxParent - 1] = temp
                idx = idxParent
            else:
                break

    # remove and return the head element (element with smallest f_cost)
    # from the heap
    def openHeap_popHead(self):
        head = self.openHeap[0]
        # Make the tail element the new head...
        self.openHeap[0] = self.openHeap[self.nOpenCells - 1]
        self.openHeapPos.reshape(-1)[self.openHeap[0]] = 1
        self.nOpenCells = self.nOpenCells - 1
        # ...and let it sink downwards until the heap property is restored
        idx = 1
        while True:
            idxPrev = idx
            idxLeft = 2 * idx  # ‼️
            idxRight = idxLeft + 1
            if idxRight <= self.nOpenCells:
                if self.f_cost.reshape(-1)[self.openHeap[idx - 1]] > self.f_cost.reshape(-1)[
                    self.openHeap[idxLeft - 1]]:
                    idx = idxLeft
                if self.f_cost.reshape(-1)[self.openHeap[idx - 1]] > self.f_cost.reshape(-1)[
                    self.openHeap[idxRight - 1]]:
                    idx = idxRight
            elif idxLeft <= self.nOpenCells:
                if self.f_cost.reshape(-1)[self.openHeap[idx - 1]] > self.f_cost.reshape(-1)[
                    self.openHeap[idxLeft - 1]]:
                    idx = idxLeft
            if idx != idxPrev:
                self.openHeapPos.reshape(-1)[self.openHeap[idx - 1]] = idxPrev
                self.openHeapPos.reshape(-1)[self.openHeap[idxPrev - 1]] = idx
                temp = self.openHeap[idx - 1]
                self.openHeap[idx - 1] = self.openHeap[idxPrev - 1]
                self.openHeap[idxPrev - 1] = temp
            else:
                break
        return head

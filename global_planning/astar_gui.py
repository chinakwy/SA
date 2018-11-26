import numpy as np
import tkinter as tk
import sys
# import time
from astar_planner import AstarPlanner


def translated_y(y):
    y_new = 300 - y
    return y_new


# initialize figure and graphic elements
window = tk.Tk()
window.title('AStar Path Planning Example')

window.wm_attributes('-topmost', 1)
windowWidth = 500
windowHeight = 400
screenWidth, screenHeight = window.maxsize()
geometryParam = '%dx%d+%d+%d' % (windowWidth, windowHeight,
                                 (screenWidth - windowWidth) / 2, (screenHeight - windowHeight) / 2)
window.geometry(geometryParam)

canvas = tk.Canvas(window, bg='white', height=windowHeight, width=windowWidth)
map_image_file = tk.PhotoImage(file='map_office.gif')
image = canvas.create_image(0, 0, anchor='nw', image=map_image_file)
canvas.place(x=40, y=40)
button_exit = tk.Button(window, text='EXIT', command=sys.exit).place(x=0, y=0)
var2 = tk.StringVar()
l2 = tk.Label(window, textvariable=var2, bg='green', font=('Arial', 12), width=10, height=1)
l2.pack()
var1 = tk.StringVar()
l1 = tk.Label(window, textvariable=var1, bg='green', font=('Arial', 12), width=10, height=1)
l1.pack()


# ======= Parameters (free to modify) =================================
start = [50, 50]
goal = [100, 150]
var2.set('goal')
goal_robot_coordinates = [goal[0], translated_y(goal[1])]
var1.set(goal_robot_coordinates)
iterations = 0
useManualStepping = True
# ======= END: Parameters =============================================

# load map
map = np.load('map_office.npy')
obstacleMap = map

sz = map.shape

# draw the start/goal point
hStart = canvas.create_oval(start[0]-3, translated_y(start[1]-3), start[0]+3, translated_y(start[1]+3), fill='green')
hGoal = canvas.create_oval(goal[0]-3, translated_y(goal[1]-3), goal[0]+3, translated_y(goal[1]+3), fill='red')


# Initialize a new planning task (if start and/or goal position has changed)
# This generates a new planner 'object'
class StartPlanner:
    def __init__(self, useManualStepping, obstacleMap, start, goal, canvas):
        self.useManualStepping = useManualStepping
        self.canvas = canvas
        self.obstacleMap = obstacleMap
        self.start = start
        self.goal = goal

        self.planner = AstarPlanner(self.obstacleMap, self.start, self.goal)
        self.colors = {
            2: 'green',  # cell in Open List
            3: 'gray',  # cell in Closed List
            # 4: 'red'  # Open Cell with min. f_cost (head of Open Heap)
        }
        self.hHeapHead = self.canvas.create_rectangle(0, 0, 0, 0, outline='red')
        self.update_map = False
        # if isManual:
        #     self.runPlanner(np.inf)
        # else:
        #     self.runPlanner(0)

    # Run at most the given number of A* iterations and update the graphics according to the results
    def runPlanner(self, iterations):
        isManual = self.useManualStepping
        if iterations < np.inf:
            [waypoints, errMsg, cellTypes, heapHead] = self.planner.run(iterations)
        else:
            # var1.set('Computing...')
            [waypoints, errMsg, cellTypes, heapHead] = self.planner.run()
            # time.sleep(1)
            # var1.set('Computing...')

        for value in [2, 3]:  # 2: 'green' cell in Open List; 3: 'gray' cell in Closed List
            draw_y, draw_x = np.where(cellTypes == value)
            for i in range(len(draw_x)):
                self.draw = self.canvas.create_line(draw_x[i], translated_y(draw_y[i]),
                                          draw_x[i] + 1, translated_y(draw_y[i] + 1), fill=self.colors[value])

        self.canvas.coords(self.hHeapHead, heapHead[0][0] - 3, translated_y(heapHead[1][0] - 3),
                                      heapHead[0][0] + 3, translated_y(heapHead[1][0] + 3))

        if len(waypoints):  # draw the best path
            for i in range(len(waypoints[0]) - 1):
                self.hPath = self.canvas.create_line(waypoints[0][i], translated_y(waypoints[1][i]),
                                           waypoints[0][i + 1], translated_y(waypoints[1][i + 1]), fill='blue')
                var1.set('')
        else:
            self.hPath = self.canvas.create_line(0, 0, 0, 0, fill='blue')
            var1.set(errMsg)

        self.update_map = True

        # setManualButtonsEnabled(isManual && isempty(waypoints) && isempty(errMsg));


sp = StartPlanner(useManualStepping, obstacleMap, start, goal, canvas)

# ======== GUI control callbacks and other internal functions =========


# ======================move goal point============================
def keyboard_move(event):
    global hGoal
    if sp.update_map == True:
        canvas.create_image(0, 0, anchor='nw', image=map_image_file)
        hStart = canvas.create_oval(start[0] - 3, translated_y(start[1] - 3), start[0] + 3, translated_y(start[1] + 3),
                                fill='green')
        hGoal = canvas.create_oval(goal[0] - 3, (goal[1] - 3), goal[0] + 3, (goal[1] + 3), fill='red')
        sp.hHeapHead = sp.canvas.create_rectangle(0, 0, 0, 0, outline='red')
    sp.update_map = False

    move_item = hGoal  # move hGoal
    move_value = 3
    canvas.itemconfig(move_item)
    if event.keysym == "Up":
        canvas.move(move_item, 0, -move_value)
        goal[1] -= move_value
    elif event.keysym == "Down":
        canvas.move(move_item, 0, move_value)
        goal[1] += move_value
    elif event.keysym == "Left":
        canvas.move(move_item, -move_value, 0)
        goal[0] -= move_value
    elif event.keysym == "Right":
        canvas.move(move_item, move_value, 0)
        goal[0] += move_value

    goal_robot_coordinates = [goal[0], translated_y(goal[1])]
    var1.set(goal_robot_coordinates)
    # var1.set(goal)
    window.update_idletasks()
    window.update()

    sp.goal = [goal[0], translated_y(goal[1])]
    # sp.goal = goal
    sp.planner = AstarPlanner(sp.obstacleMap, sp.start, sp.goal)


canvas.bind_all("<KeyPress-Up>", keyboard_move)
canvas.bind_all("<KeyPress-Down>", keyboard_move)
canvas.bind_all("<KeyPress-Left>", keyboard_move)
canvas.bind_all("<KeyPress-Right>", keyboard_move)
# ==================================================


def checkbutton_selection():
    global useManualStepping
    if checkbutton_var.get() == 0:
        useManualStepping = True
    else:
        useManualStepping = False
        sp.runPlanner(np.inf)
        hStart = canvas.create_oval(start[0] - 3, translated_y(start[1] - 3), start[0] + 3, translated_y(start[1] + 3),
                                    fill='green')  # for the start point not to be covered


checkbutton_var = tk.IntVar()
checkbutton = tk.Checkbutton(window, text="Manual Iterations:", variable=checkbutton_var, onvalue=0, offvalue=1,
                             command=checkbutton_selection).place(x=0, y=350)


def dorun():
    if useManualStepping:
        sp.runPlanner(iterations)
        hStart = canvas.create_oval(start[0] - 3, translated_y(start[1] - 3), start[0] + 3, translated_y(start[1] + 3),
                                    fill='green')  # for the start point not to be covered
        window.update_idletasks()
        window.update()


def dostep1():
    global iterations
    iterations = 1
    dorun()


def dostep10():
    global iterations
    iterations = 10
    dorun()


def dostep100():
    global iterations
    iterations = 100
    dorun()


def dostep_all():
    global iterations
    iterations = np.inf
    dorun()


button_1 = tk.Button(window, text='1', width=3, height=1, command=dostep1).place(x=150, y=350)
button_10 = tk.Button(window, text='10', width=3, height=1, command=dostep10).place(x=190, y=350)
button_100 = tk.Button(window, text='100', width=3, height=1, command=dostep100).place(x=230, y=350)
button_all = tk.Button(window, text='All', width=3, height=1, command=dostep_all).place(x=270, y=350)

if __name__ == '__main__':
    window.mainloop()



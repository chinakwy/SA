import tkinter as tk
import numpy as np
import time  # time.sleep
import sys  # sys.exit
from sensor_rangefinder2d import SensorRangefinder2d
from visual_map_scans2d import VisualMapScans2d
from visual_platform2d import VisualPlatform2d
from model_platform2d_on_path import model_platform2d_on_path, Path2d

window = tk.Tk()
window.title('Simulation: build_scan_map')
window.geometry('900x700')  # Window size
height = 600  # y
width = 800  # x
canvas = tk.Canvas(window, bg='white', height=height, width=width)  # Build a canvas
canvas.place(x=0, y=60)  # Canvas placement

var1 = tk.StringVar()
l1 = tk.Label(window, textvariable=var1, bg='green', font=('Arial', 12), width=15, height=2)
l1.pack()

# =================== main part for simulation ====================
map_image_file = tk.PhotoImage(file='./map/office.gif')
canvas.create_image(0, 0, anchor='nw', image=map_image_file)  # Add map to the canvas

pathPoints = [[1.00, 0.50],
              [1.00, 4.30],
              [2.60, 4.30],
              [2.60, 2.70],
              [6.00, 2.70],
              [6.00, 5.10],
              [7.00, 5.10],
              [6.00, 5.10],
              [6.00, 3.30],
              [7.50, 2.70],
              [7.50, 0.75],
              [5.00, 1.50],
              [2.00, 1.50],
              [2.00, 0.50]]
path = np.array(pathPoints)
X_0 = [path[0], path[1], 90*np.pi/180]
Xs = [X_0]
t = 0

platform = VisualPlatform2d(canvas)
platform.robot_radius = 0.1

path_draw = Path2d(canvas)

rangefinder = SensorRangefinder2d(canvas)
map_scans = VisualMapScans2d(canvas)


def run():
    global t, Xs

    Xs = model_platform2d_on_path(path, t, Xs)
    X = Xs[-1]
    t += 0.1

    path_draw.draw_path(t, path)
    rangefinder.draw_rays(X)
    map_scans.draw_map(X)
    platform.draw_bot(X)

    window.update_idletasks()
    window.update()
# ===============================================================


def dorun():
    while True:
        if not do_run:
            break
        var1.set('do Run')
        run()
        time.sleep(0.1)


def dopause():
    var1.set('do Pause')


do_run = False


def toggle_run_pause():
    global do_run
    if not do_run:
        do_run = True
        dorun()
    else:
        do_run = False
        dopause()


def dostep():
    global do_run
    do_run = False
    var1.set('do Step')
    run()


def dostep_back():
    global t, do_run
    do_run = False

    if t > 0.11:
        var1.set('do StepBack')

        del (Xs[-1])

        X = Xs[-1]

        rangefinder.draw_rays(X)
        platform.draw_bot(X)
        t -= 0.1
        window.update_idletasks()
        window.update()


button_img_play_green_gif = tk.PhotoImage(file='./res/play_green.gif')
button_img_pause_green_gif = tk.PhotoImage(file='./res/pause_green.gif')
button_img_next_green_gif = tk.PhotoImage(file='./res/next_green.gif')
button_img_prev_green_gif = tk.PhotoImage(file='./res/prev_green.gif')

button_doRun = tk.Button(window, image=button_img_play_green_gif, command=toggle_run_pause).place(x=40, y=10)  # run
button_doPause = tk.Button(window, image=button_img_pause_green_gif, command=toggle_run_pause).place(x=60, y=10)
button_doStep = tk.Button(window, image=button_img_next_green_gif, command=dostep).place(x=80, y=10)  # doStep
button_doStepBack = tk.Button(window, image=button_img_prev_green_gif, command=dostep_back).place(x=100, y=10)  # doBack

button_exit = tk.Button(window, text='EXIT', command=sys.exit).place(x=0, y=0)

if __name__ == '__main__':
    window.mainloop()



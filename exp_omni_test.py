import tkinter as tk
import numpy as np
import time
import sys  # sys.exit
from guidance_omni_circle import GuidanceOmniCircle
from visual_platform2d import VisualPlatform2d
from model_platform2d_omni import ModelPlatform2dOmni

window = tk.Tk()
window.title('Simulation: omni_test')
window.geometry('850x750')
height = 600  # y
width = 800  # x
canvas = tk.Canvas(window, bg='white', height=height, width=width)

canvas.place(x=0, y=60)

var1 = tk.StringVar()
l1 = tk.Label(window, textvariable=var1, font=('Verdana bold', 18))
l1.pack()

# =================== main part for simulation ====================
map_image_file = tk.PhotoImage(file='./map/emptyroom.gif')
image = canvas.create_image(0, 0, anchor='nw', image=map_image_file)

X_0 = [2, 2.5, 90*np.pi/180]
Xs = [X_0]
t = 0

controller = GuidanceOmniCircle()
controller.radius = 2
controller.rotVelocity = 20 * np.pi / 180

platform = VisualPlatform2d(canvas)
platform.robot_radius = 0.1


def run():
    global t, Xs
    if t == 0:
        X_last = Xs[-1]
    else:
        X_last = Xs[-2]
    X = Xs[-1]
    controller_output = controller.guide(t)
    model = ModelPlatform2dOmni(controller_output)
    Xs = model.continuous_integration(Xs, t)

    t += 0.1

    platform.draw_bot(X)
    platform.draw_track(X, X_last)
    window.update_idletasks()
    window.update()
# ===============================================================


def dorun():
    global t
    while True:
        if not do_run:
            break
        var1.set('Running')
        run()
        time.sleep(0.06)


def dopause():
    var1.set('Pause')


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
    global t, do_run
    do_run = False
    var1.set('Step Forward')
    run()


def dostep_back():
    global t, do_run
    do_run = False

    if t > 0.11:
        var1.set('Step Back')
        t -= 0.1

        del(Xs[-1])
        X = Xs[-1]
        X_last = Xs[-2]

        platform.draw_bot(X)
        platform.draw_track(X, X_last)
        window.update_idletasks()
        window.update()


button_img_play_green_gif = tk.PhotoImage(file='./res/play_green.gif')
button_img_pause_green_gif = tk.PhotoImage(file='./res/pause_green.gif')
button_img_next_green_gif = tk.PhotoImage(file='./res/next_green.gif')
button_img_prev_green_gif = tk.PhotoImage(file='./res/prev_green.gif')


button_doRun = tk.Button(window, image=button_img_play_green_gif, command=toggle_run_pause).place(x=60, y=10)  # run
button_doPause = tk.Button(window, image=button_img_pause_green_gif, command=toggle_run_pause).place(x=90, y=10)
button_doStep = tk.Button(window, image=button_img_next_green_gif, command=dostep).place(x=120, y=10)  # doStep
button_doStepBack = tk.Button(window, image=button_img_prev_green_gif, command=dostep_back).place(x=150, y=10)  # doBack


button_exit = tk.Button(window, text='EXIT', command=sys.exit).place(x=0, y=0)

# s = tk.Scale(window, label='timeline', from_=0, to=10, orient=tk.HORIZONTAL, length=500, showvalue=1,
#              tickinterval=5, resolution=0.01).place(x=150, y=670)

if __name__ == '__main__':
    window.mainloop()

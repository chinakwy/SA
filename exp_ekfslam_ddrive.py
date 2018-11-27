import tkinter as tk
import numpy as np
import time
import sys  # sys.exit
from visual_platform2d import VisualPlatform2d
from model_platform2d_ddrive import ModelPlatform2dDDrive
from controller_ddrive_follow_path2d import ControllerDDriveFollowPath2d  # 1／50
from sensor_landmarks2d import SensorLandmarks2d  # 1/15
from sensor_odometer_wheelspeed import SensorOdometerWheelspeed  # 1/100
from filter_ddrive_ekfslam import FilterDDriveEKFSlAM
from visual_filter_slam2d import VisualFilterSLAM2d
from visual_filter_localization2d import VisualFilterLocalization2d
from visual_error_figure import VisulaErrorFigure
from visual_cov_figure import VisualCovFigure


''' Perform Localization & SLAM experiments with a single or multiple robots
 Inputs:
  - filter algorithm: available localization algorithms: 'EKF', 'EIF', 'UKF', 'PF'|'MCL'
                      available SLAM algorithms: 'EKFSLAM'|'SLAM'
                      default: 'EKF'
  - number of robots: default = 1'''

window = tk.Tk()
window.title('Simulation: ekfslam_ddrive')
window.geometry('900x700')
window.resizable(False, False)

height = 600  # y
width = 800  # x
canvas = tk.Canvas(window, bg='white', height=height, width=width)


var1 = tk.StringVar()
l1 = tk.Label(window, textvariable=var1, font=('Verdana bold', 18))
l1.pack()

var2 = tk.StringVar()
l2 = tk.Label(window, textvariable=var2, font=('Arial bold', 12))
l2.pack()
var2.set('SLAM with Extended Kalman Filter')

# =================== main part for simulation ====================
# Some configuration options for the experiment
useEmptyRoom = False
useFilterPose = True

if useEmptyRoom:
    # Prepare map and (rectangular) path for empty-room experiment setup
    map_image_file = tk.PhotoImage(file='./map/emptyroom.gif')
    image = canvas.create_image(0, 0, anchor='nw', image=map_image_file)
    canvas.place(x=0, y=60)
    pathPoints = ([2.00, 0.75],
                  [2.00, 4.75],
                  [6.00, 4.75],
                  [6.00, 0.75],
                  [2.00, 0.75])
    path = np.array(pathPoints)

else:
    # Prepare map and path for office environment experiment setup
    map_image_file = tk.PhotoImage(file='./map/office.gif')
    image = canvas.create_image(0, 0, anchor='nw', image=map_image_file)
    canvas.place(x=0, y=60)
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

# The first path point is used as initial pose
X_0 = np.array([path[0][0], path[0][1], 90*np.pi/180])
Xs = [X_0]
current_poses = [X_0]
X = Xs[-1]
t = 0

# add the controller (which uses the 'platform' as input by default)
controller = ControllerDDriveFollowPath2d(path, canvas)
controller.wheelRadius = [0.03, 0.03]
controller.wheelDistance = 0.25


# add the sensors
odometer = SensorOdometerWheelspeed()
odometer.odometryError = 10 * np.pi / 180

landmarks_detector = SensorLandmarks2d(canvas, useEmptyRoom)
landmarks_detector.range = 7
landmarks_detector.fieldOfView = np.array([-70, 70]) * np.pi / 180
landmarks_detector.bearingError = 5 * np.pi / 180
landmarks_detector.rangeError = 2 / 100


#  ...and finally the localization filter
# (variant 1: prediction using ODE45(scipy.integrate.odeint); linearize, then discretize; use cartesian measurements)
slam = FilterDDriveEKFSlAM(odometer, landmarks_detector)
slam.useNumericPrediction = False
slam.useExactDiscretization = True

slam2d = VisualFilterSLAM2d(canvas, landmarks_detector.landmarks)

local_filter2d = VisualFilterLocalization2d(canvas)
visual_errfigure = VisulaErrorFigure(slam)
visual_covfigure = VisualCovFigure(slam)


# instantiate the platform (differential drive) and set its parameters
platform = VisualPlatform2d(canvas)
platform.robot_radius = 0.1


def run():
    global t, Xs, current_poses, xs, covs, control_outputs
    X = Xs[-1]
    if t == 0:
        control_outputs = controller.control(t, Xs)

    model = ModelPlatform2dDDrive(np.array(control_outputs['u'])[-1])  # u = [omega_r, omega_l]
    Xs = model.continuous_integration(t, Xs, 1/15)

    estimated_values = slam.filter_step(t, X, Xs, np.array(control_outputs['u'])[-1])
    covs = np.array(estimated_values['cov'])
    xs = np.array(estimated_values.pose)

    if useFilterPose:
        current_poses = xs
    else:
        current_poses = Xs
    control_outputs = controller.control(t, current_poses)

    platform.draw_bot(current_poses[-1])
    controller.draw_path(t)
    controller.draw_nextpoint()

    landmarks_detector.draw_landmarks(t)
    landmarks_detector.draw_fov(current_poses[-1])
    landmarks_detector.draw_connections(t, current_poses[-1])

    local_filter2d.draw_pose(covs[-1], xs[-1])
    local_filter2d.draw_track(t, xs)

    slam2d.draw_features(estimated_values)  # Prediction point
    slam2d.draw_covariances(estimated_values)
    slam2d.draw_lm_associations(estimated_values)

    t += 1/15
    t = round(t, 4)

    window.update_idletasks()
    window.update()
# ===============================================================


def dorun():
    while True:
        if not do_run:
            break
        var1.set('Running')
        run()
        time.sleep(0.01)


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
    global do_run
    do_run = False
    var1.set('Step Forward')
    run()


def dostep_back():
    global t, do_run, Xs, xs, covs, control_outputs, estimated_values, X
    do_run = False

    if t > 0.11:
        var1.set('Step Back')
        del (Xs[-1])
        xs = np.delete(xs, -1, axis=0)  # Delete last one row of array
        covs = np.delete(covs, -1, axis=0)
        X = Xs[-1]

        odometer.out = odometer.out[odometer.out['t'] <= t]
        landmarks_detector.out = landmarks_detector.out[landmarks_detector.out['t'] <= t]
        slam.out = slam.out[slam.out['t'] <= t]
        controller.out = controller.out[controller.out['t'] <= t]

        control_outputs = controller.out
        estimated_values = slam.out

        if useFilterPose:
            current_poses = xs
        else:
            current_poses = Xs

        platform.draw_bot(current_poses[-1])
        controller.draw_path(t)
        controller.draw_nextpoint()

        landmarks_detector.draw_fov(current_poses[-1])  # Scanning range
        landmarks_detector.draw_connections(t, current_poses[-1])

        local_filter2d.draw_pose(covs[-1], xs[-1])

        slam2d.draw_features(estimated_values)
        slam2d.draw_covariances(estimated_values)
        slam2d.draw_lm_associations(estimated_values)
        t -= 1/15
        window.update_idletasks()
        window.update()


button_img_play_green_gif = tk.PhotoImage(file='./res/play_green.gif')
button_img_pause_green_gif = tk.PhotoImage(file='./res/pause_green.gif')
button_img_next_green_gif = tk.PhotoImage(file='./res/next_green.gif')
button_img_prev_green_gif = tk.PhotoImage(file='./res/prev_green.gif')


button_img_covariance_icon_gif = tk.PhotoImage(file='./res/covariance_icon.gif')
button_img_figure_default_gif = tk.PhotoImage(file='./res/figure_default.gif')


button_doRun = tk.Button(window, image=button_img_play_green_gif, command=toggle_run_pause).place(x=60, y=10)  # run
button_doPause = tk.Button(window, image=button_img_pause_green_gif, command=toggle_run_pause).place(x=90, y=10)
button_doStep = tk.Button(window, image=button_img_next_green_gif, command=dostep).place(x=120, y=10)  # doStep
button_doStepBack = tk.Button(window, image=button_img_prev_green_gif, command=dostep_back).place(x=150, y=10)  # doBack

button_covariance = tk.Button(window, image=button_img_covariance_icon_gif,
                              command=visual_covfigure.visual_cov_figure).place(x=180, y=10)
button_error = tk.Button(window, image=button_img_figure_default_gif,
                         command=visual_errfigure.visual_error_figure).place(x=210, y=10)

button_exit = tk.Button(window, text='EXIT', command=sys.exit).place(x=0, y=0)

if __name__ == '__main__':
    window.mainloop()

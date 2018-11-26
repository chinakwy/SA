import numpy as np
import pandas as pd


# add the sensors


class SensorOdometerWheelspeed:
    def __init__(self):
        self.odometryError = 5 * np.pi / 180
        self.speeds = []
        self.ts = []
        self.sampling_time = 1 / 100  # Abtastzeit

    def sample(self, t, control_output):  # controller.control(t, Xs)
        # global wheelspeed_state
        # if not len(wheelspeed_state):
        #     wheelspeed_state = [0, 0]
        global wheelspeed_state
        if t == 0:
            wheelspeed_state = [0, 0]

        odometer_t = t  # sampling_time 1/100
        while odometer_t <= t+0.0999:
            if not len(control_output):
                speed = wheelspeed_state
            else:
                speed = control_output
                wheelspeed_state = speed
            speed = speed + np.dot(self.odometryError, np.random.randn(speed.size))

            odometer_t = round(odometer_t, 4)
            if odometer_t not in self.ts and odometer_t > 0:
                # Three identical results appear in front of the list, use this to remove the same data

                self.speeds.append(speed)
                self.ts.append(odometer_t)

            odometer_t += self.sampling_time

        data = {'t': self.ts, 'speed': self.speeds}
        self.out = pd.DataFrame(data)
        return self.out

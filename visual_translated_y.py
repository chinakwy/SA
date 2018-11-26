# The result of the calculation is all under the robot coordinates, that is, the origin is the lower left corner.
# But for display the image coordinates are used , and the top left corner is the origin.
# So the origin should be changed from the upper left corner to the lower left corner in the canvas.


def translated_y(y):
    y_new = 600 - y
    return y_new

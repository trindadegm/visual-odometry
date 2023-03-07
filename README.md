# Gyro Recorder Visualizer

This is a python prototype application used for visualizing a sensor-recording
of a game-controller's _IMU_ (Inertial Measurement Unit). The input data is a
CSV file in the following format:

```csv
<TIMESTAMP>, <A_X>, <A_Y>, <A_Z>, <G_X>, <G_Y>, <G_Z>
```

Where `TIMESTAMP` is a time-stamp in microseconds, `A_(X,Y,Z)` represents the X,
Y, and Z components of the accelerometer's reading, and `G_(X, Y, Z)` represents
the components of the gyroscope reading.

The application requires two CSV files as inputs, the first one the recording of
interest, and the second one a recording with the game-controller sitting still
to serve as calibration.
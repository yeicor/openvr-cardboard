"""OpenVR Cardboard tracking runtime.

The package is split into testable, dependency-light modules:

* ``protocol`` / ``driver_link`` - UDP protocol and OpenVR driver link.
* ``camera`` / ``markers`` / ``pose_tracking`` - webcam tracking.
* ``fusion`` / ``pipeline`` - pose composition at 250 Hz.
* ``phone_link`` / ``video`` - Android phone IMU and video streaming.
* ``gamepad`` - Xbox-style controller input for each hand.
"""

__version__ = "1.1.0"
APP_NAME = "OpenVR Cardboard"

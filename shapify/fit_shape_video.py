"""SHAPify video command line entry point.

Multi-view β fitting for static-subject / moving-camera smartphone capture.
Relaxes the T-pose requirement of ``shapify.fit_shape`` by sharing β and
body_pose across frames and refitting per-frame cameras jointly.
"""

from .runner_video import main


if __name__ == "__main__":
    main()

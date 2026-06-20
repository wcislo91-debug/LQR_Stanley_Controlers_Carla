#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints

        throttle_output = 0.0
        steer_output    = 0.0
        brake_output    = 0.0

        ######################################################
        # PERSISTENT VARIABLES
        ######################################################
        self.vars.create_var('v_error_prev', 0.0)
        self.vars.create_var('a_prev', 0.0)
        self.vars.create_var('throttle_prev', 0.0)

        # LQR gain (constant)
        self.vars.create_var('_lqr_K', np.array([1.2, 0.4]))

        if self._start_control_loop:

            ######################################################
            # LONGITUDINAL LQR CONTROLLER
            ######################################################
            dt = max(t - self.vars.t_prev, 1e-3) if hasattr(self.vars, "t_prev") else 0.05
            self.vars.t_prev = t

            # Velocity error and derivative
            v_error = v - v_desired
            v_error_dot = (v_error - self.vars.v_error_prev) / dt
            self.vars.v_error_prev = v_error

            # LQR state
            state = np.array([v_error, v_error_dot])
            a_cmd = -float(self.vars._lqr_K.dot(state))

            # Smooth and clamp
            a_cmd = np.clip(a_cmd, -1.0, 1.0)
            a_cmd = 0.8 * self.vars.a_prev + 0.2 * a_cmd
            self.vars.a_prev = a_cmd

            # Convert acceleration to throttle/brake
            if a_cmd >= 0:
                throttle_output = np.clip(a_cmd, 0.0, 1.0)
                brake_output = 0.0
            else:
                throttle_output = 0.0
                brake_output = np.clip(-a_cmd, 0.0, 1.0)

        ######################################################
        # LATERAL STANLEY CONTROLLER (FINAL FIXED VERSION)
        ######################################################

            # Extract path arrays
            path_x = np.array([wp[0] for wp in waypoints])
            path_y = np.array([wp[1] for wp in waypoints])

            # --- 1) Find nearest waypoint ---
            dx_all = path_x - x
            dy_all = path_y - y
            d2 = dx_all**2 + dy_all**2
            target_idx = int(np.argmin(d2))

            # --- 2) Compute path yaw from a valid segment ---
            # Avoid using a degenerate segment (dx=0, dy=0)
            if target_idx == 0:
                i0, i1 = 0, 1
            elif target_idx >= len(path_x) - 1:
                i0, i1 = len(path_x) - 2, len(path_x) - 1
            else:
                i0, i1 = target_idx, target_idx + 1

            seg_dx = path_x[i1] - path_x[i0]
            seg_dy = path_y[i1] - path_y[i0]

            # If segment is degenerate, force yaw = vehicle yaw (prevents random steering)
            if abs(seg_dx) < 1e-6 and abs(seg_dy) < 1e-6:
                yaw_path = yaw
            else:
                yaw_path = np.arctan2(seg_dy, seg_dx)

            # --- 3) Heading error ---
            heading_error = np.arctan2(np.sin(yaw_path - yaw),
                                    np.cos(yaw_path - yaw))

            # --- 4) Signed lateral error using path normal ---
            map_x = path_x[target_idx]
            map_y = path_y[target_idx]

            ex = x - map_x
            ey = y - map_y

            # Left-hand normal of path direction
            nx = -np.sin(yaw_path)
            ny =  np.cos(yaw_path)

            lateral_error = -(ex * nx + ey * ny)

            # --- 5) DEADZONE to prevent steering on straight path ---
            if abs(lateral_error) < 0.03 and abs(heading_error) < 0.015:
                delta = 0.0
            else:
                # --- 6) Stanley control law ---
                k = 0.8
                kv = 1.0
                v_eff = max(v, 0.5)  # avoid huge steering at low speed

                cross_term = np.arctan2(k * lateral_error, v_eff + kv)
                delta = heading_error + cross_term

            # --- 7) Clamp steering ---
            max_steer_rad = np.deg2rad(70.0)
            delta = np.clip(delta, -max_steer_rad, max_steer_rad)

            steer_output = delta



        ######################################################
        # SEND COMMANDS TO SIMULATOR
        ######################################################
        self.set_throttle(throttle_output)
        self.set_steer(steer_output)
        self.set_brake(brake_output)

        ######################################################
        # STORE OLD VALUES
        ######################################################
        self.vars.v_previous = v


#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from hippo_msgs.msg import ActuatorSetpoint, Float64Stamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf_transformations import euler_from_quaternion
import numpy as np


class YawController(Node):

    def __init__(self):
        super().__init__(node_name='yaw_controller')
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)

        # default value for the yaw setpoint
        self.setpoint = math.pi / 2.0
        self.time_before = self.get_clock().now().nanoseconds * 1e-9
        self.sum = 0.0
        self.yaw_filtered_before = math.pi/2.0
        self.error_before = 0.0
        self.start_time = self.get_clock().now()

        self.vision_pose_sub = self.create_subscription(
            msg_type=PoseWithCovarianceStamped,
            topic='vision_pose_cov',
            callback=self.on_vision_pose,
            qos_profile=qos)
        self.setpoint_sub = self.create_subscription(Float64Stamped,
                                                     topic='~/setpoint',
                                                     callback=self.on_setpoint,
                                                     qos_profile=qos)

        self.torque_pub = self.create_publisher(msg_type=ActuatorSetpoint,
                                                topic='torque_setpoint',
                                                qos_profile=1)

    def wrap_pi(self, value: float):
        """Normalize the angle to the range [-pi; pi]."""
        if (-math.pi < value) and (value < math.pi):
            return value
        range = 2 * math.pi
        num_wraps = math.floor((value + math.pi) / range)
        return value - range * num_wraps

    def on_setpoint(self, msg: Float64Stamped):
        self.setpoint = self.wrap_pi(msg.data)

    def on_vision_pose(self, msg: PoseWithCovarianceStamped):
        # get the vehicle orientation expressed as quaternion
        q = msg.pose.pose.orientation
        # convert the quaternion to euler angles
        (roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw = self.wrap_pi(yaw)

        control_output = self.compute_control_output(yaw)
        timestamp = rclpy.time.Time.from_msg(msg.header.stamp)
        self.publish_control_output(control_output, timestamp)

    def compute_control_output(self, yaw):
        # very important: normalize the angle error!
        time_now = self.get_clock().now().nanoseconds *1e-9
        dt = time_now - self.time_before
        time_since_start = time_now - self.start_time.nanoseconds*1e-9
        yaw_filtered = 0.2*yaw + 0.8*self.yaw_filtered_before
        error = self.wrap_pi(self.setpoint - yaw_filtered)
        derror = self.wrap_pi(error-self.error_before)/dt
        self.sum+=(dt*error)

        p_gain = 0.3    #turned out to be a good value
        d_gain = 0.2
        if time_since_start > 5:
            d_gain = 0.0
        i_gain = 0.005
        self.yaw_filtered_before = yaw_filtered
        self.time_before = time_now
        self.error_before = error
        return p_gain*error + d_gain*derror #+i_gain*self.sum

    def publish_control_output(self, control_output: float,
                               timestamp: rclpy.time.Time):
        msg = ActuatorSetpoint()
        msg.header.stamp = timestamp.to_msg()
        msg.ignore_x = True
        msg.ignore_y = True
        msg.ignore_z = False  #yaw is the rotation around the vehicle's z axis
        dt = (self.get_clock().now()-self.start_time).nanoseconds*1e-9
        if dt < 5:
            msg.z = 0.0
        else:
            if abs(control_output) > 0.6:
                msg.z = np.sign(control_output) * 0.6
            else:
                msg.z = control_output
        self.torque_pub.publish(msg)


def main():
    rclpy.init()
    node = YawController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
import rclpy
import numpy as np
from hippo_msgs.msg import ActuatorSetpoint, Float64Stamped, RangeMeasurementArray, RangeMeasurement
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

class PositioncontrolNode(Node):
    def __init__(self):
        super().__init__(node_name='position_control_node')
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)
        self.position_setpoint_sub = self.create_subscription(msg_type=PoseStamped, topic='xy_setpoint',
                                                callback=self.on_setpoint, qos_profile=1)
        self.position_sub = self.create_subscription(msg_type=PoseStamped, topic='position_estimate', 
                                                     callback=self.on_pose, qos_profile=1)
        self.thrust_pub = self.create_publisher(msg_type=PoseStamped, topic='xy_thrust_setpoint', #PoseStamped, xy_thrust_setpoint
                                                 qos_profile=1)
        self.ranges_sub = self.create_subscription(msg_type=RangeMeasurementArray, topic='ranges', callback=self.on_ranges,
                                                   qos_profile=qos)
        self.safety_timer_pub = self.create_publisher(msg_type=Float64Stamped, topic='safety_timer',
                                                      qos_profile=1)
        
        self.current_setpoint = np.zeros((3,1))
        self.error_before = np.zeros((3,1))
        self.time_before = self.get_clock().now()
        self.sum = np.zeros((3,1))
        self.ranges_start_timer = self.get_clock().now().nanoseconds*1e-9
        self.safety_timer = 0.0
        self.timer = self.create_timer(timer_period_sec=1 / 50,
                                       callback=self.on_timer)

    def safety_timer_publisher(self,now:rclpy.time.Time):
        msg = Float64Stamped()
        msg.data = self.safety_timer
        msg.header.stamp = now.to_msg()
        self.safety_timer_pub.publish(msg)

    def on_timer(self):
        self.safety_timer = self.get_clock().now().nanoseconds*1e-9 - self.ranges_start_timer
        #self.get_logger().info(
        #    f"safety_timer = {self.safety_timer}, start_timer ={self.ranges_start_timer}",
        #    throttle_duration_sec=2)
        now = self.get_clock().now()
        self.safety_timer_publisher(now)

    def on_ranges(self, msg:RangeMeasurementArray):
        num_measurements = len(msg.measurements)
        if num_measurements >= 2:
            self.ranges_start_timer = self.get_clock().now().nanoseconds*1e-9

    def on_setpoint(self, setpoint_msg:PoseStamped):
        self.current_setpoint[0,0] = setpoint_msg.pose.position.x
        self.current_setpoint[1,0] = setpoint_msg.pose.position.y
        self.current_setpoint[2,0] = setpoint_msg.pose.position.z

    def on_pose(self, pose_msg:PoseStamped):
        current_position = np.zeros((3,1))
        current_position[0,0] = pose_msg.pose.position.x
        current_position[1,0] = pose_msg.pose.position.y
        current_position[2,0] = pose_msg.pose.position.z
        thrust = self.compute_actuator_setpoint(current_position=current_position)
        now = self.get_clock().now()
        self.publish_thrust(thrust=thrust, now=now, current_position=current_position)

    def compute_actuator_setpoint(self, current_position:np.ndarray(shape=(3,1))) -> np.ndarray:
        error = self.current_setpoint-current_position
        derror = error - self.error_before
        current_time = self.get_clock().now()
        dt = (current_time - self.time_before).nanoseconds*1e-9
        d_part = 1.0/dt * derror
        self.sum = self.sum + dt * error

        p_gain = 2.5   #2.5 for don, 0.5 or 0.3 for doff
        d_gain = 2.5       #2.5
        d_gain_z = 2.2
        i_gain = 0.05

        thrust = np.zeros((3,1))

        thrust[0:2,0] = p_gain*error[0:2,0] + d_gain*d_part[0:2,0] #+ i_gain * self.sum[0:2,0]
        thrust[2,0] = p_gain*error[2,0] + 0.0*d_gain_z*d_part[2,0] + i_gain * self.sum[2,0]
        self.error_before = error
        self.time_before = current_time
        #self.get_logger().info(
        #    f"sum = {self.sum}",
        #    throttle_duration_sec=2)
        return thrust
    
    def publish_thrust(self, thrust:np.ndarray, now:rclpy.time.Time, current_position:np.ndarray(shape=(3,1))):
        msg = PoseStamped()

        #if current_position[0,0] < 0.2 or current_position[0,0] > 1.8:
        #    thrust[0,0] = 0.0
        #if current_position[1,0] < 0.2 or current_position[1,0] > 4.8:
        #    thrust[1,0] = 0.0
        msg.pose.position.x = thrust[0,0]
        msg.pose.position.y = thrust[1,0]
        msg.header.stamp = now.to_msg()

        self.thrust_pub.publish(msg)

def main():
    rclpy.init()
    node = PositioncontrolNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

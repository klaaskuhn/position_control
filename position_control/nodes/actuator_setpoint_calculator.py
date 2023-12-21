#!/usr/bin/env python3
from std_msgs.msg import Float64MultiArray
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from hippo_msgs.msg import Float64Stamped, ControlTarget, ActuatorSetpoint
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from scipy.spatial.transform import Rotation as R
from tf_transformations import euler_from_quaternion
import rclpy
import numpy as np

class ActuatorCalculatorNode(Node):
    def __init__(self):
        super().__init__(node_name='actuator_calculator')
        self.current_x_setpoint = 0.0
        self.current_y_setpoint = 0.0
        self.current_z_setpoint = 0.0
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)
        self.r = R.from_euler('x',[0.0])
        self.time_recent_xy_command = self.get_clock().now().nanoseconds*1e-9
        self.actuator_setpoint_publisher = self.create_publisher(msg_type=ActuatorSetpoint,
                                                                 topic='thrust_setpoint',
                                                                 qos_profile=1)
        
        self.depth_actuator_sub = self.create_subscription(msg_type=Float64Stamped,
                                                           topic='z_thrust_setpoint',
                                                           callback=self.on_z_thrust_setpoint,
                                                           qos_profile=1)
        
        self.xy_actuator_sub = self.create_subscription(msg_type=PoseStamped,
                                                        topic='xy_thrust_setpoint',
                                                        callback=self.on_xy_thrust_setpoint,
                                                        qos_profile=1)
        
        self.orientation_sub = self.create_subscription(msg_type=PoseWithCovarianceStamped, topic='vision_pose_cov',
                                                        callback=self.on_vision_pose, qos_profile=qos)
    
    def on_vision_pose(self,msg:PoseWithCovarianceStamped):
        q = msg.pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.r = R.from_euler(seq='xyz', angles=[roll, pitch, yaw],
                          degrees=False)
        
        pass
        
    def on_xy_thrust_setpoint(self, msg:PoseStamped):
        self.time_recent_xy_command = self.get_clock().now().nanoseconds*1e-9
        self.current_x_setpoint = msg.pose.position.x
        self.current_y_setpoint = msg.pose.position.y
        now = self.get_clock().now()
        self.thrust_setpoint_publish(now=now)

    def on_z_thrust_setpoint(self, msg:Float64Stamped):
        self.current_z_setpoint = msg.data
        now = self.get_clock().now()
        self.thrust_setpoint_publish(now=now)


    def thrust_setpoint_publish(self, now:rclpy.time.Time):
        msg = ActuatorSetpoint()
        dt = self.get_clock().now().nanoseconds*1e-9-self.time_recent_xy_command
        setpoint_array = np.array([self.current_x_setpoint, self.current_y_setpoint,
                                   self.current_z_setpoint])
        if dt > 0.5:
            setpoint_array[0] = 0.0
            setpoint_array[1] = 0.0
        
        for i in range(0, len(setpoint_array)):
            if abs(setpoint_array[i]) > 0.6:
                setpoint_array[i] = np.sign(setpoint_array[i]) * 0.6
        
        ##self.r.apply(vectors=setpoint_array, inverse=True)
        ##msg.x = setpoint_array[0]
        ##msg.y = setpoint_array[1]
        ##msg.z = 0.0*setpoint_array[2]
        
        
        msg.ignore_x = False
        msg.ignore_y = False
        msg.ignore_z = False
        msg.x = setpoint_array[1]
        msg.y = -setpoint_array[0]#????????????????
        msg.z = setpoint_array[2]
        msg.header.stamp = now.to_msg()
        self.actuator_setpoint_publisher.publish(msg)

def main():
    rclpy.init()
    node = ActuatorCalculatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
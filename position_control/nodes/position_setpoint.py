#!/usr/bin/env python3
from std_msgs.msg import Float64MultiArray
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from hippo_msgs.msg import Float64Stamped, ControlTarget
from geometry_msgs.msg import PoseStamped
import rclpy
import numpy as np

class PositionSetpoint(Node):
    def __init__(self):
        super().__init__(node_name='xy_setpoint_publisher')
        self.xy_setpoint__pub = self.create_publisher(msg_type=PoseStamped,
                                                   topic='xy_setpoint', qos_profile=1)
        self.timer = self.create_timer(timer_period_sec=1 / 50,
                                       callback=self.on_timer)
        
        self.start_time = self.get_clock().now().nanoseconds*1e-9
        self.n = 0
    
    def on_timer(self):
        t = self.get_clock().now().nanoseconds*1e-9 - self.start_time
        x = 0.4*np.sin(2*np.pi/30*t) + 1.0
        y = 0.4*np.cos(2*np.pi/30*t) + 2.1
        #z = 0.3*np.sin(2*np.pi/10*t) - 0.7
        z = -0.7
        #x = 1.0
        #y = 2.3

        #Rehteck:
        z = -0.7
        if t > 10.0:
             self.n+=1
             self.start_time = self.get_clock().now().nanoseconds*1e-9
        if self.n == 4:
            self.n = 0
            
        if self.n == 0:
            x = 0.8
            y = 1.8+1.0/10.0*t
        
        if self.n ==1:
            x = 0.8+0.4/10.0*t
            y = 2.8
        
        if self.n == 2:
            x = 1.2
            y = 2.8-1.0/10.0*t
        
        if self.n == 3:
            x = 1.2 - 0.4/10.0*t
            y = 1.8

        

        
        
        data = np.array([x,y,z])
        self.publish_pose_setpoint(pose=data, now=self.get_clock().now())




    def publish_pose_setpoint(self, pose:np.ndarray, now: rclpy.time.Time) -> None:
        msg = PoseStamped()
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[1]
        msg.pose.position.z = pose[2]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.xy_setpoint__pub.publish(msg)


def main():
    rclpy.init()
    node = PositionSetpoint()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()



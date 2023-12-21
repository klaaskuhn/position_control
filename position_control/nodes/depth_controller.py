#!/usr/bin/env python3
"""
This node is your depth controller.
It takes as input a current depth and a given depth setpoint.
Its output is a thrust command to the BlueROV's actuators.
"""
import rclpy
from hippo_msgs.msg import ActuatorSetpoint, DepthStamped, Float64Stamped
from rclpy.node import Node


class DepthControlNode(Node):

    def __init__(self):
        super().__init__(node_name='depth_controller')

        self.current_setpoint = 0.0
        self.integral = 0.0
        self.start_time = self.get_clock().now().nanoseconds *1e-9
        self.time_before= self.get_clock().now().nanoseconds *1e-9  #zeit bei letztem "Tick" für dt
        self.error_before = 0.0                                     #vorheriger Error für derror

        



        self.thrust_pub = self.create_publisher(msg_type=Float64Stamped,
                                                topic='z_thrust_setpoint',
                                                qos_profile=1)

        self.setpoint_sub = self.create_subscription(msg_type=Float64Stamped,
                                                     topic='depth_setpoint',
                                                     callback=self.on_setpoint,
                                                     qos_profile=1)
        
        self.kal_depth_sub = self.create_subscription(msg_type=DepthStamped,
                                                      topic='kal_depth', callback=self.on_kal_depth, 
                                                      qos_profile=1)
        
    
    def on_kal_depth(self, kal_depth_msg:DepthStamped):
        current_depth = kal_depth_msg.depth
        thrust = self.compute_control_output(current_depth=current_depth)
        timestamp = self.get_clock().now()
        self.publish_vertical_thrust(thrust=thrust, timestamp=timestamp)      

    def on_setpoint(self, setpoint_msg: Float64Stamped):
        # We received a new setpoint! Let's save it, so that we can use it as
        # soon as we receive new depth data.
        self.current_setpoint = setpoint_msg.data

    def publish_vertical_thrust(self, thrust: float,
                                timestamp: rclpy.time.Time) -> None:
        msg = Float64Stamped()
        # we want to set the vertical thrust exlusively. mask out xy-components.
        #msg.ignore_x = True
        #msg.ignore_y = True
        #msg.ignore_z = False

        msg.data = thrust

        # Let's add a time stamp
        msg.header.stamp = timestamp.to_msg()

        self.thrust_pub.publish(msg)

    def compute_control_output(self, current_depth: float) -> float:
        if current_depth <= -0.8 and current_depth >= -0.1:
            return 0.0
 
        else:
            time_now = self.get_clock().now().nanoseconds *1e-9
            time_since_start = time_now - self.start_time
            dt = time_now - self.time_before
            
            error = self.current_setpoint - current_depth
            if abs(self.integral) > 10:
                self.integral = 0
            self.integral +=  error *dt
            d_error = (error-self.error_before)
            d_part = d_error/dt
            
            p_gain = 1.0 #4
            i_gain = 0.05 #0.176
            d_gain = 3.0 #4.55
            if time_since_start < 20:
                thrust_z = p_gain*error + i_gain * self.integral # + d_gain * d_part
            else:
                thrust_z = p_gain*error + i_gain * self.integral
            self.time_before = time_now
            self.error_before = error
        
        #self.get_logger().info(
        #    f"Hi! I'm your controller running. "
        #    f'dt = {dt}, error = {error}',
        #    throttle_duration_sec=1)


        return thrust_z

def main():
    rclpy.init()
    node = DepthControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

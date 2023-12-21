#!/usr/bin/env python3
"""
This node takes as input the pressure data and computes a resulting water depth.
"""
import rclpy
from hippo_msgs.msg import DepthStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import FluidPressure
import random as rnd
import numpy as np


class DepthCalculator(Node):

    def __init__(self):
        super().__init__(node_name='depth_calculator')

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)

        self.depth_pub = self.create_publisher(msg_type=DepthStamped,
                                               topic='depth',
                                               qos_profile=1)
        self.kal_depth_pub = self.create_publisher(msg_type=DepthStamped,
                                                    topic='kal_depth',
                                                    qos_profile=1)
        self.pressure_sub = self.create_subscription(msg_type=FluidPressure,
                                                     topic='pressure',
                                                     callback=self.on_pressure,
                                                     qos_profile=qos)
        self.process_update_timer = self.create_timer(
            1.0 / 50, self.on_prediction_step_timer)
        
        self.time_last_prediction = self.get_clock().now()
        self.offset = 0.15
        self.p_umgebung = 100000
        self.rho = 1000
        self.g = 9.81
        self.num_states = 2
        
        self.depth_velocity_covariance = 0.01
        if self.num_states == 2:
            self.depth_covariance = 0.0001
            self.state = np.zeros(self.num_states).reshape((-1,1))
            self.P = 0.3 * np.eye(self.num_states)
            self.Q = np.array([[self.depth_covariance**2, 0],
                                [0, self.depth_velocity_covariance**2]])
            
        elif self.num_states == 1:
            self.depth_covariance = 0.7
            self.state = 0.0
            self.P = 0.3
            self.Q = self.depth_covariance**2

        

    def on_pressure(self, pressure_msg: FluidPressure) -> None:
        pressure = pressure_msg.fluid_pressure
        depth = self.pressure_to_depth(pressure=pressure)
        now = self.get_clock().now()
        self.publish_depth_msg(depth=depth, now=now)
        dt = (now-self.time_last_prediction).nanoseconds*1e-9
        self.prediction(dt)
        self.time_last_prediction=now
        self.update(pressure=pressure+0.0*rnd.normalvariate(0.0, 70.0))
        if self.num_states == 2:
            self.publish_kal_depth_msg(self.state[0,0])
        elif self.num_states == 1:
            self.publish_kal_depth_msg(self.state)

    def publish_depth_msg(self, depth: float, now: rclpy.time.Time) -> None:
        msg = DepthStamped()
        # Let's add a time stamp
        msg.header.stamp = now.to_msg()
        # and populate the depth field
        msg.depth = depth
        self.depth_pub.publish(msg)

    def publish_kal_depth_msg(self, kal_depth:float):
        msg = DepthStamped()
        now = self.get_clock().now()
        msg.header.stamp = now.to_msg()
        msg.depth = kal_depth + self.offset
        self.kal_depth_pub.publish(msg)

    def pressure_to_depth(self, pressure: float) -> float:
        # TODO: implement the required depth calculation
        depth = -(pressure-self.p_umgebung)/(self.rho*self.g)  
        return depth +self.offset
    
    def on_prediction_step_timer(self):
        time_now = self.get_clock().now()
        dt = (time_now-self.time_last_prediction).nanoseconds*1e-9
        self.prediction(dt)
        self.time_last_prediction = time_now
        if self.num_states == 2:
            kal_depth = self.state[0,0]
        elif self.num_states == 1:
            kal_depth = self.state
        self.publish_kal_depth_msg(kal_depth=kal_depth)


    def prediction(self, dt:float):
        f = self.get_f(dt)
        self.state = f
        F = self.get_F(dt)
        if self.num_states == 2:
            self.P = F @ self.P @ F.T + self.Q
        elif self.num_states == 1:
            self.P = self.P + self.Q

    def update(self, pressure:float):
        h = self.get_h()
        H = self.get_H()
        
        if self.num_states == 2:
            y = pressure - h
            R = np.array([(300.0)**2]).reshape((1,1))
            S = H @ self.P @ H.T + R
            K = self.P @ H.T * np.linalg.inv(S)
            self.state = self.state + K@y
            self.P = (np.eye(2) - K@H)@ self.P
        elif self.num_states == 1:
            y = -(pressure-self.p_umgebung)/(self.rho*self.g)-self.state
            R = 10.0**2
            K = (self.P)/(self.P + R)
            self.state = self.state + K*y
            self.P = (1-K)*self.P

    def get_F(self, dt:float):
        if self.num_states == 2:
            F = np.array([[1, dt],
                        [0, 1]])
        elif self.num_states == 1:
            F = 1.0
        return F
    
    def get_f(self, dt:float):
        if self.num_states == 2:
            f = np.zeros(2).reshape((2,1))
            f[0,0] = self.state[0,0]+dt*self.state[1,0]
            f[1,0] = self.state[1,0]
        elif self.num_states == 1:
            f = self.state
        return f
    
    def get_h(self):
        if self.num_states == 2:
            h = -self.rho * self.g * self.state[0,0] + self.p_umgebung
            h = np.array(h).reshape((1,-1))
        elif self.num_states == 1:
            h = -self.rho * self.g * self.state + self.p_umgebung
        return h
    
    def get_H(self):
        if self.num_states == 2:
            H = np.array([-self.rho*self.g, 0]).reshape((1,2))
        elif self.num_states == 1:
            H = -self.rho*self.g
        return H



def main():
    rclpy.init()
    node = DepthCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

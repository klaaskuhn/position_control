#!/usr/bin/env python3
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from hippo_msgs.msg import RangeMeasurement, RangeMeasurementArray
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from tf_transformations import euler_from_quaternion


class PositionKalmanFilter(Node):

    def __init__(self):
        super().__init__(node_name='position_kalman_filter')

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)

        #self.init_params()

        self.time_last_prediction = self.get_clock().now()

        self.position_offset = np.array([0.0, 0.0, 0.0]).reshape((3,1))                                  #[0.2,0,0.1]

        # TODO Assuming state consists of position x,y,z -> Feel free to add
        # more!
        self.num_states = 6 


        # initial state
        self.x0 = np.zeros((self.num_states, 1))

        # state, this will be updated in Kalman filter algorithm
        self.state = np.copy(self.x0)

        # initial state covariance - how sure are we about the state?
        # TODO initial state covariance is tuning knob
        # dimension: num states x num states
        # matrix needs to be positive definite and symmetric
        self.P0 = 0.3 * np.eye(self.num_states)

        # state covariance, this will be updated in Kalman filter algorithm
        self.P = self.P0

        if self.num_states == 6:
            self.process_noise_position_stddev = 0.001
            self.process_noise_velocity_stddev = 0.05
            self.Q = np.array([[self.process_noise_position_stddev**2, 0, 0, 0, 0, 0],
                            [0, self.process_noise_position_stddev**2, 0, 0, 0, 0],
                            [0, 0, self.process_noise_position_stddev**2, 0, 0, 0],
                            [0, 0, 0, self.process_noise_velocity_stddev**2, 0, 0],
                            [0, 0, 0, 0, self.process_noise_velocity_stddev**2, 0],
                            [0, 0, 0, 0, 0, self.process_noise_velocity_stddev**2]])
        elif self.num_states == 3:
            self.process_noise_position_stddev = 0.03
            self.Q = self.process_noise_position_stddev**2*np.eye(3)

        self.range_noise_stddev = 0.12#0.12   #3state:0.12S 6state:0.05
        self.tag_poses = np.array([[0.7, 3.8, -0.5], [1.3, 3.8, -0.5],          
                                   [0.7, 3.8, -0.9], [1.3, 3.8, -0.9]])
        
        #[[0.7, 3.8, -0.5], [1.3, 3.8, -0.5],
         #[0.7, 3.8, -0.9], [1.3, 3.8, -0.9]]

        self.position_pub = self.create_publisher(msg_type=PoseStamped,
                                                  topic='position_estimate',
                                                  qos_profile=1)

        self.ranges_sub = self.create_subscription(
            msg_type=RangeMeasurementArray,
            topic='ranges',
            callback=self.on_ranges,
            qos_profile=qos)
        self.vision_pose_sub = self.create_subscription(
            msg_type=PoseWithCovarianceStamped,
            topic='vision_pose_cov',
            callback=self.on_vision_pose,
            qos_profile=qos)
        # do prediction step with 50 Hz
        self.process_update_timer = self.create_timer(
            1.0 / 50, self.on_prediction_step_timer)

    def on_ranges(self, ranges_msg: RangeMeasurementArray) -> None:
        # how many tags are detected?
        num_measurements = len(ranges_msg._measurements)

        # if no tags are detected, stop here
        if not num_measurements:
            return

        measurement: RangeMeasurement

        tag_ids = np.zeros((num_measurements,1), dtype=int)
        measurements = np.zeros((num_measurements,1))
        for index, measurement in enumerate(ranges_msg.measurements):
            tag_id = measurement.id          
            tag_ids[index,0] = tag_id
            measured_distance = measurement.range            
            measurements[index,0] = measured_distance

        now = self.get_clock().now()
        dt = (now - self.time_last_prediction).nanoseconds * 1e-9
        self.prediction(dt)
        self.time_last_prediction = now

        self.measurement_update(num_measurements, tag_ids, measurements)
        self.publish_pose_msg(state=np.copy(self.state), now=now)

    def on_vision_pose(self, msg: PoseWithCovarianceStamped):
        q = msg.pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        # TODO

    def on_prediction_step_timer(self):
        now = self.get_clock().now()
        dt = (now - self.time_last_prediction).nanoseconds * 1e-9

        self.prediction(dt)
        self.time_last_prediction = now

        # publish the estimated pose with constant rate
        self.publish_pose_msg(state=np.copy(self.state), now=now)

    def measurement_update(self, num_measurements, tag_ids, measurements):
        h = self.get_h(num_measurements, tag_ids)
        H = self.get_H(num_measurements, tag_ids)
        y = measurements - h
        R = (self.range_noise_stddev**2) * np.eye(num_measurements)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(self.num_states) - K @ H) @ self.P
    
    def prediction(self, dt: float):
        self.state = self.get_f(dt)
        F = self.get_F(dt)
        self.P = F @ self.P @ F.T + self.Q #* dt

    def publish_pose_msg(self, state: np.ndarray, now: rclpy.time.Time) -> None:
        msg = PoseStamped()

        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "map"
        state[0:3,0] = self.state[0:3,0] + self.position_offset[0:3,0]
        msg.pose.position.x = state[0, 0]
        msg.pose.position.y = state[1, 0]
        msg.pose.position.z = state[2, 0]

        self.position_pub.publish(msg)
    
    def get_h(self, num_measurements, tag_ids):
        h = np.zeros((num_measurements,1))
        for n,id in np.ndenumerate(tag_ids): #iteriere über Tags, erwarteten d-Wert in Abhängigkeit von x bestimmen
            h[n[0],0] = np.sqrt((self.state[0,0]-self.tag_poses[id,0])**2+ 
                           (self.state[1,0]-self.tag_poses[id,1])**2+
                           (self.state[2,0]-self.tag_poses[id,2])**2)
        return h
    
    def get_H(self, num_measurements, tag_ids):
        H = np.zeros((num_measurements,self.num_states))
        for j, id in np.ndenumerate(tag_ids):                    #j Zeilen, k Spalten
            for k in range(0,self.num_states):
                if k <3:
                    H[j[0],k] = (self.state[k,0] - self.tag_poses[int(id),k])/np.sqrt((self.state[0,0]-self.tag_poses[id,0])**2+
                                                                            (self.state[1,0]-self.tag_poses[id,1])**2+
                                                                            (self.state[2,0]-self.tag_poses[id,2])**2)
                else:
                    H[j[0],k] = 0.0
        return H
    
    def get_f(self, dt):
        if self.num_states == 6:
            f = np.array([[self.state[0,0] + dt*self.state[3,0]],
                        [self.state[1,0] + dt*self.state[4,0]],
                        [self.state[2,0] + dt*self.state[5,0]],
                        [self.state[3,0]],
                        [self.state[4,0]],
                        [self.state[5,0]]
                        ])
        elif self.num_states == 3:
            f = np.array([[self.state[0,0]],
                          [self.state[1,0]],
                          [self.state[2,0]]])
        return f

    
    def get_F(self, dt):
        if self.num_states == 6:
            F = np.array([[1, 0, 0, dt, 0, 0],
                        [0, 1, 0, 0, dt, 0],
                        [0, 0, 1, 0, 0, dt],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]
                        ])
        elif self.num_states == 3:
            F = np.eye(3)
        return F

def main():
    rclpy.init()
    node = PositionKalmanFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

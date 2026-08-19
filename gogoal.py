import math
import numpy as np
import matplotlib.pyplot as plt

class OmniRobot:
    def __init__(self, pos,ori=0):
        self.wheel_radius = 2.5/100
        self.robot_radius = 15/100 
        self.position = pos
        self.orientation = ori 
        self.max_yaw_velocity = 1 
        self.max_wheel_speed = 15  
        self.pat = []
        
    def move_to_target(self,velocity):
        if np.linalg.norm(velocity) > 0:
            desired_velocity = np.array([velocity[0], velocity[1], 0])  # Convert to [vx, vy, 0]
            wheel_velocities_percent = self.inverse_kinematics(desired_velocity)
            return wheel_velocities_percent
        else:
            return [0, 0, 0]
        
    def move(self,target_position, max_speed = 1.0, dt = 0.1):
        direction = target_position - self.position
        distance  = np.linalg.norm(direction)
        desired_velocity = direction / distance * min(max_speed, distance / dt)

        # Project current velocity onto desired direction (to reduce drift)
        global_velocity = self.forward_kinematics(self.inverse_kinematics([desired_velocity[0], desired_velocity[1], 0]))

        # Find error (drift correction)
        correction = desired_velocity - global_velocity

        # Apply correction
        corrected_velocity = desired_velocity + 0.5 * correction  # 50% correction factor

        # Prevent rotation
        corrected_velocity = list(corrected_velocity)
        corrected_velocity.append(0)

        # Compute wheel speeds
        wheel_velocities_percent = self.inverse_kinematics(corrected_velocity)


    def inverse_kinematics(self, velocity):

        velocity = np.array([velocity[0], velocity[1], 0])
        
        R = self.robot_radius

        jacobian = (1/self.wheel_radius) * np.array([
        [-math.sin(self.orientation), math.cos(self.orientation), R],
        [-math.sin(self.orientation+2*math.pi/3), math.cos(self.orientation+2*math.pi/3), R],
        [-math.sin(self.orientation-2*math.pi/3), -math.cos(self.orientation-2*math.pi/3), R]
        ])
        wheel_velocities = np.dot(jacobian, velocity)
        wheel_velocities_percent = (wheel_velocities / self.max_wheel_speed) * 100
        if max(abs(wheel_velocities_percent)) > 100:
            wheel_velocities_percent /= (max(abs(wheel_velocities_percent))/30)//1
            
        return wheel_velocities_percent



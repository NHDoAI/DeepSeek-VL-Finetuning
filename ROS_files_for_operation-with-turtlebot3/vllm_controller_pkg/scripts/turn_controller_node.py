#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from tf.transformations import euler_from_quaternion

class TurnController:
    def __init__(self):
        rospy.init_node('turn_controller_node')
        rospy.loginfo("Initializing TurnController...")

        # command flags
        self.turning_flag = False
        self.step_moving_flag = False
        self.continuous_moving_flag = False
        self.stop_flag = False
        self.lane_changing_flag = False
        self.lane_change_state = 0 # 0: idle, 1: first turn, 2: straight, 3: second turn
        self.lane_change_direction = 0.0
        
        # PID constants
        self.Kp = 2.0
        self.Ki = 0.1
        self.Kd = 0.2

        # Add max integral windup prevention
        self.max_integral = 1.0

        # PID variables
        self.error_sum = 0.0
        self.last_error = 0.0
        self.target_angle = 0.0
        self.current_angle = 0.0


        # Linear motion variables
        self.real_bot = rospy.get_param('~real_bot', False)

        if not self.real_bot:
            self.max_linear_speed = 0.2  # m/s
            self.cruise_speed = 0.05
            self.step_speed = 0.15
            self.step_distance = 0.45
        else:
            self.max_linear_speed = 0.05  # m/s
            self.cruise_speed = 0.025
            self.step_speed = 0.05
            self.step_distance = 0.1
        rospy.loginfo(f"Real bot mode: {self.real_bot}, Max linear speed: {self.max_linear_speed}")

        self.current_speed = 0.0     # Current speed
        self.target_speed=0.0        # initialize Target speed
        self.stop_distance = 0
        self.linear_direction = 0

        self.target_distance = 0.0
        
        self.odom_update_rate = 30.0 # Hz
        self.acceleration_rate = (self.max_linear_speed/self.odom_update_rate)      # m/s^2
        self.deceleration_rate = (self.max_linear_speed/self.odom_update_rate)     # m/s^2
        self.start_position = None
        self.current_position = None


        # Publishers and Subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.action_status_pub = rospy.Publisher('/action_status', Bool, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.command_sub = rospy.Subscriber('/turn_command', String, self.command_callback)
        rospy.loginfo("TurnController initialized and ready.")

    def publish_status(self, status):
        """Publishes the action completion status."""
        rospy.loginfo(f"--- Publishing Action Status: {status} ---")
        self.action_status_pub.publish(Bool(status))

    def command_callback(self, msg):
        rospy.loginfo(f"--- Received Command: '{msg.data}' ---")
        if msg.data == "turn_right":
            self.start_turn(-1.0)  # Turn right 90 degrees
        elif msg.data == "turn_left":
            self.start_turn(1.0)  # Turn left 90 degrees
        elif msg.data == "turn_around":
            self.start_turn(2.0)  # Turn 180 degrees
        elif msg.data == "step_forward":
            self.start_step_motion(self.step_distance)  # Move forward 0.5m
        elif msg.data == "step_backward":
            self.start_step_motion(-1*self.step_distance)  # Move backward 0.5m
        elif msg.data == "straight_forward":
            self.start_continuous_motion(self.max_linear_speed)
        elif msg.data == "slow_cruise":
            self.start_continuous_motion(self.cruise_speed)
        elif msg.data == "stop":
            self.start_stopmotion()
        elif msg.data == "change_lane_left":
            self.start_lane_change(True)  # True for left lane change
        elif msg.data == "change_lane_right":
            self.start_lane_change(False)  # False for right lane change

    def odom_callback(self, msg):
        # Get orientation from odometry
        orientation = msg.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.current_angle = yaw
        # Get position
        self.current_position = msg.pose.pose.position

        if self.turning_flag:
            rospy.loginfo(f"Current: {math.degrees(self.current_angle):.2f}°, " +
                         f"Target: {math.degrees(self.target_angle):.2f}°," +
                         f" Error: {math.degrees(self.target_angle - self.current_angle):.2f}°")
            
            self.update_pid()

        elif self.step_moving_flag:
            rospy.loginfo_throttle(1, f"Stepping: Speed: {self.current_speed:.2f} m/s, Dist: {self.target_distance:.2f}m")
            self.update_step_speed()

        elif self.continuous_moving_flag:
            #rospy.loginfo("Continuous motion")
            self.move_continuous()

        elif self.stop_flag:
            rospy.loginfo_throttle(1, f"Stopping: Speed: {self.current_speed:.2f} m/s")
            self.stop_motion()
        
        elif self.lane_changing_flag:
            rospy.loginfo_throttle(1, f"Lane Changing: State {self.lane_change_state}")
            self.update_lane_change()


    def start_turn(self, direction):
        rospy.loginfo("Action: Start Turn")
        self.turning_flag = True
        self.step_moving_flag = False
        self.continuous_moving_flag = False
        self.stop_flag = False
        self.target_angle = self.current_angle + (math.pi/2.0 * direction)
        self.error_sum = 0.0
        self.last_error = 0.0

    def update_pid(self):
        error = self.target_angle - self.current_angle
        
        # Normalize error to [-π, π]
        while error > math.pi:
            error -= 2.0 * math.pi
        while error < -math.pi:
            error += 2.0 * math.pi

        # Update integral term with anti-windup
        self.error_sum += error
        self.error_sum = max(min(self.error_sum, self.max_integral), -self.max_integral)

        error_diff = error - self.last_error
        control_effort = self.Kp * error + self.Ki * self.error_sum + self.Kd * error_diff
        self.last_error = error

        cmd_vel = Twist()
        cmd_vel.angular.z = control_effort

        # Limit angular velocity
        cmd_vel.angular.z = max(min(cmd_vel.angular.z, 0.75), -0.75)

        self.cmd_vel_pub.publish(cmd_vel)

        # More stringent completion check
        if abs(error) < 0.01 and abs(error_diff) < 0.005:  # About 0.57 degrees and checking if still moving
            rospy.loginfo("Action: Turn Complete")
            cmd_vel.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_vel)
            self.turning_flag = False
            if not self.lane_changing_flag:
                self.publish_status(True)

    def start_step_motion(self, distance):
        rospy.loginfo(f"Action: Start Step Motion (Distance: {distance}m)")
        self.turning_flag = False
        self.step_moving_flag = True
        self.continuous_moving_flag = False
        self.stop_flag = False
        self.target_distance = distance
        self.start_position = self.current_position
        self.target_speed = self.step_speed

    def accelerate(self, current_speed, accel_rate):
        linear_speed = current_speed + accel_rate
        return min(linear_speed, self.target_speed)

    def decelerate(self, current_speed, decel_rate):
        linear_speed = current_speed - decel_rate
        return max(linear_speed, self.target_speed)

    def update_step_speed(self):
        if self.start_position is None or self.current_position is None:
            return

        # Calculate distance moved
        dx = self.current_position.x - self.start_position.x
        dy = self.current_position.y - self.start_position.y
        distance_moved = math.sqrt(dx**2 + dy**2)

        # Determine direction based on target
        self.linear_direction = 1.0 if self.target_distance > 0 else -1.0
        self.stop_distance = (self.current_speed * (self.current_speed//self.deceleration_rate)*(1/self.odom_update_rate))/2
        if distance_moved < (abs(self.target_distance)-self.stop_distance):
            cmd_vel = Twist()
            self.current_speed = self.accelerate(self.current_speed, self.acceleration_rate)
            cmd_vel.linear.x = self.linear_direction * self.current_speed
            #print(f"current speed: {self.current_speed:.2f}m/s")
            self.cmd_vel_pub.publish(cmd_vel)
            rospy.loginfo(f"Distance moved: {distance_moved:.2f}m")
        else:
            print("start stop motion")
            self.start_stopmotion()

    def start_stopmotion(self):
        rospy.loginfo("Action: Start Stop Motion")
        self.turning_flag = False
        self.step_moving_flag = False
        self.continuous_moving_flag = False
        self.stop_flag = True
        self.target_speed = 0.0
    def stop_motion(self):
        """         dx = self.current_position.x - self.start_position.x
        dy = self.current_position.y - self.start_position.y
        distance_moved = math.sqrt(dx**2 + dy**2)
        rospy.loginfo(f"Distance moved: {distance_moved:.2f}m") """

        cmd_vel = Twist()
        self.current_speed = self.decelerate(self.current_speed, self.deceleration_rate)
        cmd_vel.linear.x = self.linear_direction * self.current_speed
        cmd_vel.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd_vel)
        if self.current_speed <= 0:
            rospy.loginfo("Motion stopped completely.")
            self.current_speed = 0
            self.linear_direction = 0
            self.stop_flag = False
            rospy.loginfo("Motion stopped")
            # Only publish status if it's not part of a lane change
            if not self.lane_changing_flag:
                self.publish_status(True)

    def start_continuous_motion(self, target_speed=None):
        rospy.loginfo(f"Action: Start Continuous Motion (Target Speed: {target_speed}m/s)")
        self.turning_flag = False
        self.step_moving_flag = False
        self.continuous_moving_flag = True
        self.stop_flag = False
        
        # Set default target speed if none provided
        if target_speed is None:
            target_speed = self.max_linear_speed
        
        # Ensure target speed is positive and within limits
        self.target_speed = min(abs(target_speed), self.max_linear_speed)
        
        # If target speed is too low, just stop
        if self.target_speed < self.deceleration_rate:
            self.start_stopmotion()
            return
        
        # Set direction to forward
        self.linear_direction = 1.0

    def move_continuous(self):
        cmd_vel = Twist()
        
        # Compare current speed with target speed, accelerate if lower, decelerate if higher, does not modify if equal
        # if abs(self.current_speed) < self.target_speed:
        #     # Need to accelerate
        #     self.current_speed = self.accelerate(self.current_speed, self.acceleration_rate)
        # elif abs(self.current_speed) > self.target_speed:
        #     # Need to decelerate
        #     self.current_speed = self.decelerate(self.current_speed, self.deceleration_rate)
        
        # Apply direction
        cmd_vel.linear.x = self.linear_direction * self.target_speed # Simpler logic for now
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Log current and target speeds
        rospy.loginfo_throttle(1, f"Continuous move: Target: {self.target_speed:.2f}")

    def start_lane_change(self, is_left):
        """Start the lane change sequence using a state machine."""
        rospy.loginfo(f"--- Starting Lane Change: {'left' if is_left else 'right'} ---")
        self.turning_flag = False
        self.step_moving_flag = False
        self.continuous_moving_flag = False
        self.stop_flag = False
        
        self.lane_changing_flag = True
        self.lane_change_state = 1
        self.lane_change_direction = 1.0 if is_left else -1.0
        
        # Start the first turn
        self.start_turn(self.lane_change_direction) # Turn 90 degrees

    def update_lane_change(self):
        """State machine to handle the lane change sequence."""
        if self.lane_change_state == 1: # Waiting for the first turn to complete
            if not self.turning_flag:
                rospy.loginfo("Lane change state: 1 -> 2 (First turn complete)")
                self.lane_change_state = 2
                self.start_step_motion(self.step_distance) # Move sideways

        elif self.lane_change_state == 2: # Waiting for the straight move to complete
            if not self.step_moving_flag and not self.stop_flag:
                rospy.loginfo("Lane change state: 2 -> 3 (Sideways move complete)")
                self.lane_change_state = 3
                self.start_turn(-self.lane_change_direction) # Turn back 90 degrees

        elif self.lane_change_state == 3: # Waiting for the second turn to complete
            if not self.turning_flag:
                rospy.loginfo("Lane change state: 3 -> 0 (Second turn complete)")
                self.lane_changing_flag = False
                self.lane_change_state = 0
                self.publish_status(True) # Lane change is fully complete


if __name__ == '__main__':
    try:
        controller = TurnController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 
    except Exception as e:
        rospy.logerr(f"Unhandled exception in TurnController: {e}") 
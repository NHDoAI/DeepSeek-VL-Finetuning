#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
import rospy
from std_msgs.msg import String

class TurtlebotControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Turtlebot3 Control Panel")
        
        # Initialize ROS node
        rospy.init_node('control_gui_node', anonymous=True)
        self.command_pub = rospy.Publisher('/turn_command', String, queue_size=10)
        
        # Command mapping
        self.command_map = {
            "step forward": "step_forward",
            "step backward": "step_backward",
            "turn left": "turn_left",
            "turn right": "turn_right",
            "turn around": "turn_around",
            "straight forward": "straight_forward",
            "slow cruise": "slow_cruise",
            "stop": "stop",
            "switch lane": "switch_lane",
        }
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create and arrange buttons
        # Movement controls
        self.create_button(main_frame, "Step Forward", "step forward", 0, 1)
        self.create_button(main_frame, "Step Backward", "step backward", 2, 1)
        self.create_button(main_frame, "Turn Left", "turn left", 1, 0)
        self.create_button(main_frame, "Turn Right", "turn right", 1, 2)
        self.create_button(main_frame, "Turn Around", "turn around", 1, 1)
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # Continuous movement and stop
        self.create_button(main_frame, "Straight Forward", "straight forward", 4, 1)
        self.create_button(main_frame, "Stop", "stop", 5, 1, bg='red')
        self.create_button(main_frame, "Slow Cruise", "slow cruise", 5, 0)
        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=6, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # Lane change controls
        self.create_button(main_frame, "Switch Lane", "switch lane", 7, 1)
        
        # Add padding to all children
        for child in main_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)
            
    def create_button(self, parent, text, command, row, col, bg=None):
        button = tk.Button(
            parent,
            text=text,
            width=15,
            height=2,
            command=lambda: self.send_command(command)
        )
        if bg:
            button.configure(bg=bg)
        button.grid(row=row, column=col)
        
    def send_command(self, command):
        msg = String()
        # Use the command map to translate to the correct format
        msg.data = self.command_map[command]
        self.command_pub.publish(msg)
        rospy.loginfo(f"GUI Command sent: {command} → {msg.data}")

def main():
    root = tk.Tk()
    app = TurtlebotControlGUI(root)
    
    # Handle ROS shutdown
    def on_closing():
        root.quit()
        rospy.signal_shutdown('GUI closed')
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main() 
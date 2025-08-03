#!/home/ubuntu/required_pkgs/vllm_demo/bin/python3

"""
ROS node for running inference with a DeepSeek-VL model.
This node uses a main loop to fetch data on demand, process it,
and publish a command, ensuring no processing backlog or GUI threading issues.
"""

import os
import sys
import threading
import rospy
import torch
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
import numpy as np
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from deepseek_vl.models import VLChatProcessor
from rapidfuzz import fuzz
import cv2
import math

def extract_phrase(text_target, phrase_query, score_cutoff=0):
    """Extract a phrase from target text using fuzzy matching."""
    split_target = text_target.split()
    split_query = phrase_query.split()
    query_word_count = len(split_query)
    target_word_count = len(split_target)
    best_score = 0
    best_match = ""
    for i in range(target_word_count + 1 - query_word_count):
        candidate = " ".join(split_target[i:i+query_word_count])
        score = fuzz.ratio(candidate, phrase_query, score_cutoff=score_cutoff)
        if score > 0.0 and score > best_score:
            best_score = score
            best_match = candidate
    return best_match, best_score

def detect_category(text_target, categories_query, score_cutoff=75):
    """Detect category from text using fuzzy matching."""
    best_score = 0
    detected_category = "N/a"
    for category in categories_query:
        score = fuzz.partial_ratio(text_target, category, score_cutoff=score_cutoff)
        if score > 0.0 and score > best_score:
            best_score = score
            detected_category = category
    return detected_category, best_score


class VLLMInferenceNode:
    def __init__(self):
        rospy.init_node('vllm_inference_node')
        rospy.loginfo("Initializing VLLMInferenceNode...")

        # --- Configuration ---
        self.model_path = rospy.get_param('~model_path', "/home/ubuntu/DeepSeek-VL-Finetuning/training_stage/prod_test/b6-s322_short-lidar_dropout-0.15_loss_4bit/")
        self.command_topic = rospy.get_param('~command_topic', '/turn_command')
        self.status_topic = rospy.get_param('~status_topic', '/action_status')
        self.real_bot = rospy.get_param('~real_bot', False)
        self.use_compressed = rospy.get_param('~use_compressed', False)
        self.last_known_lane = rospy.get_param('~initial_lane', 'left lane')
        self.visualize = rospy.get_param('~visualize', False)
        
        self.image_topic_name = '/raspicam_node/image' if self.real_bot else '/turtlebot3/camera/image_raw'
        if self.use_compressed:
            self.image_topic_name += '/compressed'

        # --- Prompts ---
        if self.real_bot:
            self.prompt_template = "<image_placeholder> Analyze the given image from a real toy vehicle's front-facing camera (The real toy vehicle travels at half the speed of the simulated vehicle). First determine the lane the vehicle is on. Then determine if there is an obstacle directly ahead in the lane the vehicle is currently occupying or not. If an obstacle is present, determine how far away the obstacle is and decide the best movement option based on that. The lidar points are as follows: {} Guidelines for Decision Making: First determine if the vehicle is currently on the right lane or left lane of the road. Then determine if no obstacle is detected in the current lane or if it is 'far away', then the vehicle should continue moving 'straight forward'. If an obstacle is detected in the current lane (not the other lane) and it is 'near' to the vehicle, it should 'slow cruise' cautiously until the vehicle is very close to the obstacle. Then if an obstacle is detected on the current lane and it is 'very close' to the vehicle (there is little to no road left between the vehicle and the obstacle), the vehicle should 'switch lane'. Response Format (Strict Adherence Required): Lane: {{left lane | right lane | unclear}}; Obstacles: {{not on the same lane | far away | near | very close}}; Decision: {{decision}}"
        else:
            self.prompt_template = "<image_placeholder> Analyze the given image from a simulated vehicle's front-facing camera. First determine the lane the vehicle is on. Then determine if there is an obstacle directly ahead in the lane the vehicle is currently occupying or not. If an obstacle is present, determine how far away the obstacle is and decide the best movement option based on that. The lidar points are as follows: {} Guidelines for Decision Making: First determine if the vehicle is currently on the right lane or left lane of the road. Then determine if no obstacle is detected in the current lane or if it is 'far away', then the vehicle should continue moving 'straight forward'. If an obstacle is detected in the current lane (not the other lane) and it is 'near' to the vehicle, it should 'slow cruise' cautiously until the vehicle is very close to the obstacle. Then if an obstacle is detected on the current lane and it is 'very close' to the vehicle (there is little to no road left between the vehicle and the obstacle), the vehicle should 'switch lane'. Response Format (Strict Adherence Required): Lane: {{left lane | right lane | unclear}}; Obstacles: {{not on the same lane | far away | near | very close}}; Decision: {{decision}}"

        # --- ROS specific initializations ---
        self.bridge = CvBridge()
        # Event to signal when a discrete action is running and we should wait
        self.action_in_progress = threading.Event()
        self.action_in_progress.clear() # Start with no action running

        # --- Latest sensor data storage ---
        self.latest_cv_image = None
        self.image_lock = threading.Lock()
        self.latest_lidar_scan = None
        self.lidar_lock = threading.Lock()
        
        # --- Load Model and Processor ---
        self.load_model()

        # --- Subscribers and Publishers ---
        image_msg_type = CompressedImage if self.use_compressed else Image
        self.image_sub = rospy.Subscriber(self.image_topic_name, image_msg_type, self.image_callback, queue_size=1)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback, queue_size=1)
        self.command_pub = rospy.Publisher(self.command_topic, String, queue_size=10)
        self.status_sub = rospy.Subscriber(self.status_topic, Bool, self.status_callback, queue_size=1)
        
        rospy.on_shutdown(self.cleanup)
        rospy.loginfo("VLLMInferenceNode initialized and ready.")

    def image_callback(self, msg):
        """Store the latest image from the camera."""
        with self.image_lock:
            try:
                if self.use_compressed:
                    cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                else:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                if self.real_bot:
                    # Rotate the image 180 degrees if it's from the real bot's camera
                    cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
                
                self.latest_cv_image = cv_image
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")

    def lidar_callback(self, msg):
        """Store the latest lidar scan."""
        with self.lidar_lock:
            self.latest_lidar_scan = msg

    def status_callback(self, msg):
        if msg.data:
            rospy.loginfo("--- Action Complete Signal Received ---")
            self.action_in_progress.clear() # Signal that the discrete action has finished
        else:
            rospy.logwarn("Received 'False' on status topic. Still waiting.")

    def get_lidar_data_string(self):
        # (This function remains largely the same)
        rospy.loginfo("Waiting for a single LiDAR scan...")
        with self.lidar_lock:
            msg = self.latest_lidar_scan
        
        if msg is None:
            rospy.logwarn("No LiDAR scan available yet. Using N/A.")
            return "{ 179.0: N/A ; 180.0: N/A ; 181.0: N/A }"

        target_angles = {179: "N/A", 180: "N/A", 181: "N/A"}
        found_angles = set()
        angle_increment = msg.angle_increment
        current_angle_rad = msg.angle_min

        for r in msg.ranges:
            if current_angle_rad > msg.angle_max: break
            angle_deg = math.degrees(current_angle_rad)
            angle_int = int(round(angle_deg))

            if angle_int in target_angles and angle_int not in found_angles:
                if math.isinf(r): target_angles[angle_int] = "inf"
                else: target_angles[angle_int] = f"{r:.4f}"
                found_angles.add(angle_int)
            
            current_angle_rad += angle_increment
            if len(found_angles) == len(target_angles): break
        
        parts = [f"{float(k)}: {v}" for k, v in sorted(target_angles.items())]
        return f"{{ { ' ; '.join(parts)} }}"

    def load_model(self):
        # (This function remains the same)
        rospy.loginfo(f"Loading model from {self.model_path}...")
        try:
            self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.tokenizer = self.vl_chat_processor.tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            rospy.loginfo("Chat processor and tokenizer loaded.")
        except Exception as e:
            rospy.logerr(f"Error loading chat processor: {e}")
            sys.exit(1)
        
        rospy.loginfo(f"Loading 4-bit model from {self.model_path} for inference...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        try:
            self.vl_gpt = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto"
            )
            self.vl_gpt = self.vl_gpt.eval()
            rospy.loginfo("Model loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Error loading model from {self.model_path}: {e}")
            sys.exit(1)

    def run_inference_cycle(self):
        """The main processing function called from the main loop."""
        # Do not start a new cycle if a discrete action is still in progress
        if self.action_in_progress.is_set():
            #rospy.loginfo_throttle(2, "Waiting for discrete action to complete...")
            return

        #rospy.loginfo("--- Starting New Inference Cycle ---")
        try:
            # Step 1: Get latest image from callback
            with self.image_lock:
                if self.latest_cv_image is None:
                    #rospy.loginfo_throttle(1, "Waiting for first image...")
                    return
                cv_image = self.latest_cv_image.copy()
            
            # Step 2: Get latest LiDAR from callback
            lidar_string = self.get_lidar_data_string()

            # Step 3: Process and visualize image
            if self.visualize:
                cv2.imshow("VLLM Inference Input", cv_image)
                # This waitKey is now in the main thread!
                cv2.waitKey(1)

            # Step 4: Run inference
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            answer = self.run_inference(pil_image, lidar_string)

            # Step 5: Publish result and update state if necessary
            command = self.process_and_publish(answer)

            # If we issued a command for a discrete, long-running action,
            # set the flag so we wait for the completion signal.
            # These are actions for which the controller WILL send a status update.
            discrete_actions = ["change_lane_left", "change_lane_right", "turn_left", 
                                "turn_right", "turn_around", "step_forward", "step_backward"]
            if command in discrete_actions:
                #rospy.loginfo(f"Discrete command '{command}' issued. Waiting for completion signal...")
                self.action_in_progress.set()
            # For all other commands (straight, cruise, stop), we can proceed to the next cycle immediately.

        except Exception as e:
            rospy.logerr(f"An error occurred during inference cycle: {e}")

    def run_inference(self, pil_image, lidar_data_for_prompt):
        # (This function remains the same, but now returns the answer)
        #rospy.loginfo("--- Running Inference ---")
        final_prompt = self.prompt_template.format(lidar_data_for_prompt)
        conversation = [{"role": "User", "content": final_prompt, "images": ["dummy_path"]}, {"role": "Assistant", "content": ""}]

        prepare_inputs = self.vl_chat_processor(conversations=conversation, images=[pil_image], force_batchify=True).to(self.vl_gpt.device)
        prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(torch.float16)
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        with torch.no_grad():
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                max_new_tokens=50,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id
            )
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        #rospy.loginfo(f"Generated Answer: {answer.strip()}")
        return answer.strip()

    def process_and_publish(self, answer):
        # (This function remains the same, but now returns the command)
        #rospy.loginfo("--- Processing and Publishing ---")
        lane_options = ["left lane", "right lane", "unclear"]
        best_score_for_lane_extraction = -1
        best_matched_phrase_for_lane = ""
        for option in lane_options:
            current_dynamic_query_lane = f"Lane: {option}"
            temp_matched_phrase, temp_score = extract_phrase(answer.lower(), current_dynamic_query_lane.lower())
            if temp_score > best_score_for_lane_extraction:
                best_score_for_lane_extraction = temp_score
                best_matched_phrase_for_lane = temp_matched_phrase
        detected_lane, _ = detect_category(best_matched_phrase_for_lane.lower(), lane_options, score_cutoff=50)

        if detected_lane in ["left lane", "right lane"]:
            self.last_known_lane = detected_lane
            #rospy.loginfo(f"Last known lane updated to: '{self.last_known_lane}'")

        decision_options = ["straight forward", "slow cruise", "switch lane"]
        best_score_for_decision_extraction = -1
        best_matched_phrase_for_decision = ""
        for option in decision_options:
            current_dynamic_query_decision = f"Decision: {option}"
            temp_matched_phrase, temp_score = extract_phrase(answer.lower(), current_dynamic_query_decision.lower())
            if temp_score > best_score_for_decision_extraction:
                best_score_for_decision_extraction = temp_score
                best_matched_phrase_for_decision = temp_matched_phrase
        detected_decision, _ = detect_category(best_matched_phrase_for_decision.lower(), decision_options, score_cutoff=50)

        rospy.loginfo(f"Detected Lane: {detected_lane}, Detected Decision: {detected_decision}")
        
        command = "stop" # Default command
        if detected_decision == "straight forward": command = "straight_forward"
        elif detected_decision == "slow cruise": command = "slow_cruise"
        elif detected_decision == "switch lane":
            if self.last_known_lane == "left lane": command = "change_lane_right"
            elif self.last_known_lane == "right lane": command = "change_lane_left"
            else:
                rospy.logwarn(f"Cannot switch lane, last known lane is not set clearly ('{self.last_known_lane}'). Stopping.")
                command = "stop"
        
        command_msg = String()
        command_msg.data = command
        self.command_pub.publish(command_msg)
        #rospy.loginfo(f"Published command: '{command}'")
        return command # Return the command for the main loop

    def cleanup(self):
        rospy.loginfo("Shutting down and closing windows.")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = VLLMInferenceNode()
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            node.run_inference_cycle()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"An unhandled exception occurred in the main loop: {e}")
    finally:
        cv2.destroyAllWindows() 
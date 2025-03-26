from main import Main
import time
import random
from traffic_lights import TrafficLights
import os
import numpy as np


class Train:
    def __init__(self, generations, end_count):
        self.main_instance = Main()
        self.generations = generations
        self.end_count = end_count
        self.reward_dic = {}
        self.rewards = []

    def reset_environment(self):
        self.main_instance.current_light_state = "RED"
        self.main_instance.starting_traffic_light = random.choice(
            self.main_instance.traffic_light_parameters["directions"])
        self.main_instance.traffic_lights = TrafficLights(
            self.main_instance.screen,
            self.main_instance.starting_traffic_light,
            self.main_instance.current_light_state,
            self.main_instance.traffic_light_parameters["directions"],
            self.main_instance.colors["traffic_lights"],
            self.main_instance.traffic_light_width,
            self.main_instance.intersection_center,
            self.main_instance.road_width,
            self.main_instance.intersection_trl_width,
            self.main_instance.traffic_light_parameters["timings"],
        )

        # Clear all vehicles and reset related parameters
        with self.main_instance.vehicle_list_lock:
            self.main_instance.vehicle_list.clear()
        self.main_instance.vehicle_parameters["vehicle_count"] = {"north": 0, "south": 0, "east": 0, "west": 0}
        self.main_instance.vehicle_parameters["processed_vehicles"] = {"north": 0, "south": 0, "east": 0, "west": 0}
        self.main_instance.vehicle_parameters["dti_info"] = {"north": {}, "south": {}, "east": {}, "west": {}}

        # Reset timers and counters
        self.main_instance.last_action_time = None

        # Reset reward and other metrics
        self.main_instance.total_reward = 0

        self.main_instance.initial_epsilon = 0.9

        self.main_instance.reward_list = []
        self.main_instance.total_reward = 0

    def save_model(self):
        # Ensure the directory for saving exists
        os.makedirs("saved_models", exist_ok=True)
        # Save the Q-table
        np.save("saved_models/sarsa_q_table.npy", self.main_instance.sarsa_agent.q_table)
        print("Model saved successfully.")

    def calculate_accuracy(self, current_reward):
        """
        Calculate accuracy as the percentile rank of the current reward
        compared to all past rewards.
        """
        if not self.rewards:
            return 0  # Avoid calculation before rewards exist

        # Sort rewards and find percentile rank of current_reward
        sorted_rewards = sorted(self.rewards)
        rank = sum(1 for r in sorted_rewards if r <= current_reward)
        accuracy = (rank / len(sorted_rewards)) * 100
        return round(accuracy, 2)  # Ensure accuracy stays within 0-100%


    def train(self):
        for generation in range(self.generations):
            self.reset_environment()
            self.main_instance.initialize_sarsa()
            total_reward = self.main_instance.run(generation + 1, True, self.end_count)

            # Track rewards for normalization
            self.rewards.append(total_reward)
            self.reward_dic[generation] = total_reward

            # Calculate accuracy as percentile rank
            accuracy = self.calculate_accuracy(total_reward)

            # Display reward and accuracy for this generation
            print(f"Generation: {generation + 1} | Reward: {total_reward} | Accuracy: {accuracy:.2f}%")
            time.sleep(1)

        self.save_model()


if __name__ == "__main__":
    train_model = Train(generations=100, end_count=8)

    start_time = time.time()
    train_model.train()  # Assuming train executes all generations
    total_time = time.time() - start_time

    print(f"Total time for 10 generations: {total_time:.2f} seconds")
    print(f"Average time per generation: {total_time / 10:.2f} seconds")

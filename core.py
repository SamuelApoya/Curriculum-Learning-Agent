import random
import numpy as np
import tensorflow as tf

# Each Task represents a learning activity with a set of concepts and a difficulty level
class Task:
    def __init__(self, task_id: str, concepts: list, difficulty: float = 1.0):
        self.task_id = task_id  # Unique task identifier
        self.concepts = concepts  # List of concept names the task involves
        self.difficulty = difficulty  # Difficulty affects how much the student learns from it


# A simple student model that updates its concept knowledge based on tasks
class StudentModel:
    def __init__(self, concept_list: list, learning_rate: float = 0.1):
        self.concept_list = concept_list  # The list of all possible concepts
        self.learning_rate = learning_rate  # Scalar factor controlling rate of knowledge increase
        self.reset()

    def reset(self):
        # Initializes student knowledge to 0 for all concepts
        self.knowledge = {concept: 0.0 for concept in self.concept_list}

    def get_state(self) -> np.ndarray:
        # Returns the student's current knowledge as a fixed-size vector
        return np.array([self.knowledge[c] for c in self.concept_list], dtype=np.float32)

    def train_on_task(self, task: Task) -> float:
        # Simulates learning from a task: updates knowledge based on difficulty and learning rate
        improvement = 0.0
        for concept in task.concepts:
            before = self.knowledge[concept]
            self.knowledge[concept] = min(1.0, before + self.learning_rate * task.difficulty)
            improvement += self.knowledge[concept] - before
        return improvement  # Used as the reward signal

    def evaluate(self, tasks: list) -> float:
        # Computes average mastery over the concepts in the given tasks
        if not tasks:
            return 0.0
        total_score = 0.0
        for task in tasks:
            scores = [self.knowledge.get(c, 0.0) for c in task.concepts]
            total_score += np.mean(scores)
        return total_score / len(tasks)


# CurriculumAgent chooses tasks using a neural network to approximate Q-values (DQN)
class CurriculumAgent:
    def __init__(self, concept_list, task_pool, lr=0.001, gamma=0.9, epsilon=0.2):
        self.concept_list = concept_list  # Used for input size of the neural network
        self.task_pool = task_pool  # List of all available tasks (actions)
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration probability for epsilon-greedy policy
        self.num_tasks = len(task_pool)  # Total number of available actions
        self.task_id_to_idx = {task.task_id: i for i, task in enumerate(task_pool)}  # Map task ID to index

        # Simple neural network: input = state vector, output = Q-values for each task
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(len(concept_list),)),  # Input is student's knowledge vector
            tf.keras.layers.Dense(64, activation='relu'),       # Hidden layer
            tf.keras.layers.Dense(self.num_tasks)               # Q-value for each task
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def select_task(self, state_vec: np.ndarray) -> Task:
        # Chooses a task using epsilon-greedy policy: either explore or exploit
        if random.random() < self.epsilon:
            return random.choice(self.task_pool)  # Explore
        q_values = self.model(np.expand_dims(state_vec, axis=0))[0].numpy()  # Predict Q-values
        best_idx = np.argmax(q_values)  # Choose task with max predicted Q-value
        return self.task_pool[best_idx]

    def update_q(self, state, action_task, reward, next_state, done):
        """
        Performs a DQN-style update:
        - state: current state vector
        - action_task: task taken
        - reward: immediate reward received
        - next_state: resulting state vector
        - done: whether the episode is complete
        """
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        action_index = self.task_id_to_idx[action_task.task_id]

        with tf.GradientTape() as tape:
            q_values = self.model(state)  # Q(s, ·)
            target_q_values = q_values.numpy().copy()

            # Estimate max_a Q(s', a) for next state
            next_q_values = self.model(next_state)[0].numpy()
            max_next_q = np.max(next_q_values)

            # Compute TD target
            target = reward if done else reward + self.gamma * max_next_q
            target_q_values[0][action_index] = target

            # Compute loss between predicted Q and target Q
            loss = self.loss_fn(target_q_values, q_values)

        # Update model weights using backprop
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

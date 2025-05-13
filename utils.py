import numpy as np
import matplotlib.pyplot as plt
from core import Task

def generate_synthetic_tasks(concepts, num_tasks=10, min_concepts=1, max_concepts=3):
    """
    Generates synthetic tasks with random concepts and difficulty levels.
    """
    tasks = []
    for i in range(num_tasks):
        num_c = np.random.randint(min_concepts, max_concepts + 1)
        chosen = list(np.random.choice(concepts, size=num_c, replace=False))
        difficulty = np.random.uniform(0.5, 1.5)
        task = Task(task_id=f"T{i}", concepts=chosen, difficulty=difficulty)
        tasks.append(task)
    return tasks


def evaluate_on_target(student, target_tasks):
    """
    Evaluates the student's average performance on the target tasks.
    """
    return student.evaluate(target_tasks)


def format_state_dict(state_dict):
    """
    Returns a readable version of the student's state for logging.
    """
    return {k: f"{v:.2f}" for k, v in state_dict.items()}


def plot_learning_curve(rewards, label='Reward', title='Learning Curve'):
    """
    Plots the learning curve for the curriculum agent.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(rewards, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_concept_mastery(knowledge_dict, title="Student Knowledge State"):
    """
    Plots the student's current concept mastery as a bar chart.
    """
    concepts = list(knowledge_dict.keys())
    values = list(knowledge_dict.values())
    plt.figure(figsize=(8, 4))
    plt.bar(concepts, values, color='skyblue')
    plt.ylim(0, 1.0)
    plt.xlabel("Concepts")
    plt.ylabel("Mastery")
    plt.title(title)
    plt.tight_layout()

import streamlit as st
import numpy as np
from core import Task, StudentModel, CurriculumAgent
from utils import generate_synthetic_tasks, evaluate_on_target, plot_learning_curve, plot_concept_mastery
import matplotlib.pyplot as plt

# Setup
concepts = ["algebra", "geometry", "probability", "calculus", "statistics"]
task_pool = generate_synthetic_tasks(concepts, num_tasks=15)
target_tasks = []

# State initialization
if "student" not in st.session_state:
    st.session_state.student = StudentModel(concept_list=concepts)
    st.session_state.agent = CurriculumAgent(concepts, task_pool)
    st.session_state.rewards = []
    st.session_state.evaluations = []
    st.session_state.episodes_run = 0

# UI
st.title("Curriculum Learning Agent Dashboard")
st.markdown("Simulates a tutor that uses DQN to optimize a student's learning path.")

# Target selection
selected_targets = st.multiselect("Select Target Concepts", concepts, default=concepts[:2])
target_tasks = [task for task in task_pool if any(c in selected_targets for c in task.concepts)]

# Run episodes
num_episodes = st.slider("How many episodes to run?", 1, 50, 10)
if st.button("Run Episodes"):
    for _ in range(num_episodes):
        student = st.session_state.student
        agent = st.session_state.agent
        student.reset()
        state = student.get_state()
        total_reward = 0

        for _ in range(10):  # Steps per episode
            task = agent.select_task(state)
            reward = student.train_on_task(task)
            total_reward += reward
            next_state = student.get_state()
            agent.update_q(state, task, reward, next_state, done=False)
            state = next_state

        st.session_state.rewards.append(total_reward)
        eval_score = evaluate_on_target(student, target_tasks)
        st.session_state.evaluations.append(eval_score)
        st.session_state.episodes_run += 1

# Plots
if st.session_state.rewards:
    st.subheader("Episode Rewards")
    fig, ax = plt.subplots()
    ax.plot(st.session_state.rewards, label="Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Over Episodes")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🎓 Target Evaluation")
    fig, ax = plt.subplots()
    ax.plot(st.session_state.evaluations, label="Target Eval", color='orange')
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Score")
    ax.set_title("Target Evaluation Over Time")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Final Concept Mastery")
    plot_concept_mastery(st.session_state.student.knowledge, title="Current Student Knowledge")
    st.pyplot(plt.gcf())

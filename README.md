# Official Implementation for "Harmonizing Parameterized Verification and Invariant Synthesis for Safety Envelopes in Black-Box Autonomy"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official source code, experimental setup, and data for the research paper: **"Harmonizing Parameterized Verification and Invariant Synthesis for Safety Envelopes in Black-Box Autonomy"**.

Our work introduces a novel framework to ensure the safety of black-box autonomous agents, particularly those powered by Large Language Models (LLMs), when deployed in safety-critical, parameterized systems. Instead of attempting the often-impossible task of verifying the agent's internal logic, we synthesize and enforce a **Safety Envelope**—a set of verifiable, runtime-enforced constraints that provably guarantee adherence to high-level safety specifications.

[**➡️ Read the Full Paper (Link to be added here when available, e.g., arXiv or publisher's site)**]

## Framework Overview

The core challenge we address is the "Verification Gap" where a powerful but opaque agent's decisions could lead to safety violations. Our framework bridges this gap by introducing a verifiable intermediary layer.

![Verification Gap](https://raw.githubusercontent.com/kuangjinjun/Safety-Envelopes/main/assets/fig_verification_gap.png)

Our approach consists of:
1.  **Formal Modeling**: Representing the system family as a Parameterized Transition System (PTS).
2.  **Invariant Synthesis**: Deriving parameterized invariants that guarantee safety across all system configurations.
3.  **Safety Envelope Synthesis**: Automatically learning a computationally simple, yet effective, safety envelope from a high-level specification, even for black-box agents.
4.  **Runtime Monitoring & Enforcement (RME)**: A mechanism that checks the agent's proposed actions against the envelope *before* execution, preventing unsafe actions.

## Experimental Testbed

The code in this repository implements the resource scheduling testbed described in the paper. The environment simulates an agent managing a task queue with parameterized capacity (`Q_max`) and facing maliciously crafted "trap tasks" designed to induce safety violations.

We evaluate several agent configurations:
- **Baseline Agents (BA)**: Unconstrained LLM-based agents.
- **Heuristic Agent (HA)**: An agent with a hard-coded, expert-designed safety logic.
- **Our Monitored Agents (OMA)**: LLM-based agents guarded by our synthesized Safety Envelope.

## Getting Started

### Prerequisites

The experiments are implemented in Python 3. You will need to install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/kuangjinjun/Safety-Envelopes.git
cd Safety-Envelopes

# Install dependencies
pip install -r requirements.txt

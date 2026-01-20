# BubbleRAN Agentic Toolkit - Agent Development Kit (ADK)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/bat-adk)](https://pypi.org/project/bat-adk/)

The **BAT-ADK** is a Python-based Software Development Kit designed to simplify the development, deployment, and integration of AI Agents within the BubbleRAN architecture.
This repository includes the ADK framework ([BubbleRAN Software License](https://bubbleran.com/resources/files/BubbleRAN_Licence-Agreement-1.3.pdf)).

## Key Features
- 🛠️ Easy-to-use Python SDK for developing AI Agents
- 🔗 Integrates the [LangGraph](https://pypi.org/project/langgraph/) library with the [A2A SDK](https://pypi.org/project/a2a-sdk/) and [MCP SDK]() for building AI Agents beyond POCs (ready for production)
- ☁️ Ready for Cloud-Native deployment with BubbleRAN [MX-AI](https://bubbleran.com/products/mx-ai/)
- 🧩 Prebuilt Agentic Workflow (e.g. ReAct, A2A Communication)

## Getting Started

### Prerequisites
- Python 3.12+
- `uv` (recommended) or `pip`

## Installation

### Using `uv`
```bash
uv add bat-adk
```

### Using `pip`
```bash
pip install bat-adk
```

## Documentation

The BAT-ADK uses [`pydoc-markdown`](https://pydoc-markdown.readthedocs.io/) to generate API documentation directly from Python docstrings.

### Generating the Documentation

To build the documentation locally, run:

```bash
uv run pydoc-markdown
```

The generated documentation will be available at `adk/build/docs/content/bat-adk`

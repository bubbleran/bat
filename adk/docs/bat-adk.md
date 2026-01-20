# High Level documentation of BAT-ADK

## Introduction

The **BR-ADK (BubbleRAN Agent Development Kit)** is a Python SDK built on top of [LangGraph](https://www.langchain.com/langgraph) that helps you build intelligent AI agents with minimal effort. It natively supports two standard protocols:

- **A2A (Agent-to-Agent)**: for communication between agents  
- **MCP (Model Context Protocol)**: for accessing external tools, prompts, and contextual data

### Protocols at a Glance

- [A2A](https://a2a-protocol.org/latest/) is the **default and recommended** protocol for agent-to-agent communication.
- [MCP](https://modelcontextprotocol.io/docs/getting-started/intro) is used by agents to securely interact with external resources such as tools and prompts.
- While MCP *can* be used for agent-to-agent communication, this is mainly intended for compatibility with other agent frameworks. Whenever possible, prefer **A2A** for agent interactions.

### Design Philosophy

BR-ADK is designed to let you focus on **agent behavior and workflows**, not protocol details.  
You define your agent’s logic using graph-based workflows, while the SDK handles communication, protocol compliance, and integration under the hood.



## Preliminary Concepts

Before using **BR-ADK**, you should be familiar with a few core ideas:

- **LangGraph basics**  
  Understand how to define a graph, manage graph state, use streaming, and handle interrupts for **Human-in-the-Loop (HITL)** interactions.

- **A2A fundamentals**  
  Know the basics of the **A2A protocol**, especially what an **Agent Card** is and its role in agent communication.

- **MCP fundamentals**  
  Have a basic understanding of the **Model Context Protocol (MCP)** and how it enables agents to access tools, prompts, and context.



## Agent Application

An **Agent Application** is the main object for building agents in **BR-ADK**. It is an instance of the `AgentApplication` class and requires two parameters:

1. **Graph type** – a class extending `AgentGraph`  
2. **State type** – a class extending `AgentState`

### What Happens on Instantiation

When you create an `AgentApplication`, it automatically:

- Loads the **Agent Card** from `./agent.json` and the **Agent Configuration** from `./config.yaml`
- Instantiates the **AgentGraph**
- Sets up the **AgentExecutor**, request handler, and a **Starlette** web application

### Running the Agent Application

After creation, call the `run()` method to start the Starlette app and expose the **A2A Server**.  

- Use `run(expose_mcp=True)` to also start an **MCP Server**, which provides two tools:
  - `get_agent_card()` – returns the **Agent Card** as a JSON string  
  - `call_agent(query, context_id, message_id)` – sends a request to the A2A Server and returns the response  

> **Note:** All MCP requests are internally forwarded to the A2A endpoint.

### Ports

- **A2A:** 9900 (default)  
- **MCP:** 9800 (default)  

You can override these using the `PORT` and `MCP_PORT` environment variables.



## Agent Configuration

The **AgentConfig** defines how an agent behaves and what external resources it can access, including:

- Whether to perform **checkpoints**  
- Which **MCP Servers** and other **Agents** it needs to communicate with

### Loading and Validation

- Automatically loaded from `config.yaml` or the path set in the `CONFIG` environment variable when the **AgentApplication** starts  
- Validated by checking connectivity to all `required` MCP Servers and Agents  
  - If any required resource is unreachable, the application crashes and must be restarted (handled automatically in Kubernetes)

### Features

The `AgentConfig` class provides methods to:

- List available MCP Servers and Agents  
- Retrieve the **AgentCard** of specific Agents  
- Retrieve the **Tools** provided by specific MCP Servers  

These methods help you integrate external resources directly into your agent’s **Graph** logic.

### Naming Guidelines

- In the configuration, you assign **a name** and a **URL** for each MCP Server or Agent.  
- **Always use the Official name** of the server or agent in your agent logic, not the name you assigned in the config.  
- The SDK automatically maps your assigned names to the Official names during validation.  
- Using Official names ensures your agent will work correctly when deployed with tools like the **AIFabric controller** of the **Odin Operator**.

**Example:**  
```yaml
remote-agents:
  - name: MyHelperAgent   # your assigned name
    url: http://helper:9900
```
In your agent code, reference this agent by its Official name, e.g., HelperAgent, not MyHelperAgent.



## Agent Executor

The **Agent Executor** handles task execution and event publishing for an agent.  

### Key Functions

- **Execute a request**  
  - Calls the `astream` method of the **AgentGraph**.
  - Processes each chunk individually, converting `AgentTaskResult` objects into **A2A events** for the event queue.
  - Collects usage metadata from the graph at each step and includes them in the stream chunks.

- **Cancel a request**  
  - Currently **not implemented**



## Agent Graph

An **AgentGraph** defines the core logic of an agent using the **LangGraph** library. Agents built with BR-ADK stream their responses as **AgentTaskResult** objects, produced after executing each node in the graph.

### Creating a Custom Graph

You **cannot instantiate `AgentGraph` directly**. Instead:

1. **Extend** the `AgentGraph` class  
2. Implement the `setup(config: AgentConfig)` method. Inside this method:
   - Instantiate all **ChatModelClient** instances your agent will use **as properties of the extended class** (e.g. `self.<client-name>`).
   - Instantiate any prebuilt workflows.
   - Define the graph’s nodes and edges via the `graph_builder` property.

After `setup` completes, the **AgentGraph** is compiled by the BR-ADK (you don't need to do it manually) and ready to use.

### Streaming Responses

The `AgentGraph` class provides an `astream` method, which the **Agent Executor** uses to submit requests and receive streamed responses.



## Agent State

Each **AgentGraph** has a **State** that updates after each node executes. The state is defined by extending the `AgentState` class, which is a **Pydantic model** ensuring type safety.

### Implementing a Custom State

When you extend `AgentState`, you must:

- Override `from_query(str)` – to initialize the state from a query  
- Override `to_task_result()` – to convert the state into an `AgentTaskResult`  

Optional overrides for advanced features:

- `update_after_checkpoint_restore(str)` – for **multi-turn conversations**  
- `is_waiting_for_human_input()` – for **Human-in-the-Loop (HITL)** interactions

### How State Works

- Each node in the **Agent Graph** receives the current `AgentState` and returns an updated `AgentState`  
- The **Agent Graph** converts the updated state into an **AgentTaskResult** using `to_task_result()` in the `astream` method.
- The **Agent Executor** then processes the `AgentTaskResult` and generates **A2A events** for the event queue



## Chat Model Client

A **Chat Model Client** is a wrapper around an LLM that combines:

- A LangChain **BaseChatModel**
- **System instructions**
- Optional **tools**

### What It Does

- Provides `invoke` for single requests and `batch` for parallel requests  
- Tracks usage metadata:
  - Input, output, and total tokens
  - LLM inference time

When a `ChatModelClient` is created as part of an **AgentGraph**, the graph automatically collects this usage data after each node execution and includes it in the streamed results.

### Configuration

A **ChatModelClient** is configured using a **ChatModelClientConfig**, typically loaded from environment variables via `from_env`.  
The configuration includes:

- Model provider and model name  
- Optional LLM endpoint URL  
- Optional client name (developer-defined identifier)



## Prebuilt Workflows

**Prebuilt Workflows** are reusable **Runnable** components that can be used as nodes inside an **AgentGraph**.

### Purpose

They help you encapsulate common or complex logic into reusable building blocks, making agent graphs easier to design and maintain.

### Creating a Prebuilt Workflow

BR-ADK provides the abstract `PrebuiltWorkflow` class. To create one, you must extend it and implement:

- `_setup()`  
  - Similar to `AgentGraph.setup`  
  - Used to initialize models, tools, and internal state

- `_astream()`  
  - Implements the workflow’s **custom streaming logic**  
  - This is more advanced and requires careful handling

### Using a Prebuilt Workflow

Once defined, call `as_runnable()` to obtain a **Runnable** that can be directly used as a node in an **Agent Graph**.



## ReAct Loop

The **ReAct Loop** is a prebuilt workflow that implements the classic **ReAct** pattern (Reason + Act) using:

- An **LLM node**
- A **Tool node**

It lets you add a full ReAct workflow to an **Agent Graph** as a **single node**.

### How to Use It

To instantiate a ReAct Loop, you must provide:

- The **Agent State schema** used by your graph (the `AgentState` subclass you defined)
- A `ChatModelClient` with tools enabled (created inside the `AgentGraph.setup` method)
- The names of specific **state fields**:
  - Where the loop reads its input from
  - Where the loop writes its output to

Check the documentation for the full list of available parameters.

This design keeps the ReAct logic reusable while cleanly integrating with your agent’s state and graph.



## Call Agent Node

The **Call Agent Node** is a prebuilt workflow that abstracts the **Agent-to-Agent (A2A)** streaming.

It lets you integrate a remote agent call into your graph as a **single node**, handling the stream consumption and state updates automatically. This feature allows for real-time streaming of what the called agent is doing.

### How to Use It

To instantiate a Call Agent Node, you must provide:
- The **Agent Config** and **Agent State** schema
- The target agent's name (the agent must be specified in the **Agent Config**)
- A Message Builder function to construct the request from the current state
- The names of some specific **state fields**:
  - Where to store the remote agent's status and content
  - Where to flag if the remote agent needs human input

This node also automatically captures **token usage** from the remote agent and merges it into your local metrics.

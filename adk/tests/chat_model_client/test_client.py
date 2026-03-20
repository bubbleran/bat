import pytest
import time
from bat.chat_model_client import ChatModelClient, ChatModelClientConfig, UsageMetadata
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from unittest.mock import MagicMock

# ------------------ UsageMetadata Tests ------------------

def test_usage_metadata_add_instance():
    meta1 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15, inference_time=1.5)
    meta2 = UsageMetadata(input_tokens=2, output_tokens=3, total_tokens=5, inference_time=0.5)

    result = meta1 + meta2
    assert result.input_tokens == 7
    assert result.output_tokens == 13
    assert result.total_tokens == 20
    assert abs(result.inference_time - 2.0) < 1e-6

def test_usage_metadata_add_dict():
    meta1 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15, inference_time=1.5)
    meta2 = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5, "inference_time": 0.5}

    result = meta1 + meta2
    assert result.input_tokens == 7
    assert result.output_tokens == 13
    assert result.total_tokens == 20
    assert abs(result.inference_time - 2.0) < 1e-6

def test_usage_metadata_sub_instance():
    meta1 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15, inference_time=1.5)
    meta2 = UsageMetadata(input_tokens=2, output_tokens=3, total_tokens=5, inference_time=0.5)

    result = meta1 - meta2
    assert result.input_tokens == 3
    assert result.output_tokens == 7
    assert result.total_tokens == 10
    assert abs(result.inference_time - 1.0) < 1e-6

def test_usage_metadata_sub_dict():
    meta1 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15, inference_time=1.5)
    meta2 = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5, "inference_time": 0.5}

    result = meta1 - meta2
    assert result.input_tokens == 3
    assert result.output_tokens == 7
    assert result.total_tokens == 10
    assert abs(result.inference_time - 1.0) < 1e-6

def test_usage_metadata_non_negative_validator():
    with pytest.raises(ValueError):
        UsageMetadata(input_tokens=-1)
    with pytest.raises(ValueError):
        UsageMetadata(output_tokens=-1)
    with pytest.raises(ValueError):
        UsageMetadata(total_tokens=-1)
    with pytest.raises(ValueError):
        UsageMetadata(inference_time=-0.1)

# ------------------ ChatModelClient Tests ------------------

@pytest.fixture
def mock_chat_model(monkeypatch):
    mock_model = MagicMock()
    mock_model.invoke = MagicMock(return_value=AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 1,"output_tokens": 2,"total_tokens": 3},
    ))

    def batch_side_effect(messages_list):
        # Return an AIMessage for each input message
        return [
            AIMessage(
                content=f"batch ai response #{i}",
                usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
            )
            for i, _ in enumerate(messages_list, start=1)
        ]

    mock_model.batch = MagicMock(side_effect=batch_side_effect)
    monkeypatch.setattr("bat.chat_model_client.client.init_chat_model", lambda **kwargs: mock_model)
    return mock_model

@pytest.fixture
def mock_chat_model_client_config():
    return ChatModelClientConfig(
        model="test-model",
        model_provider="ollama",
        base_url="http://localhost"
    )

@pytest.fixture
def client(mock_chat_model, mock_chat_model_client_config):
    return ChatModelClient(
        chat_model_config=mock_chat_model_client_config,
        system_instructions="system instructions",
    )

def test_system_instructions_type_error():
    with pytest.raises(TypeError):
        ChatModelClient(system_instructions=['Instruction 1', 'Instruction 2'])

def test_validate_input_type(client):
    assert client._validate_input_type("string input")
    assert client._validate_input_type(HumanMessage("human message input"))
    assert client._validate_input_type([ToolMessage(
        tool_call_id="id-123",
        content="tool_call_result",
    )])
    assert not client._validate_input_type([HumanMessage("list of human message inputs")])

def test_build_messages_list_with_human_message(client):
    human_msg = HumanMessage("hello")
    history = [AIMessage("prev")]
    msgs = client._build_messages_list(human_msg, history)
    assert msgs[0] == client.system_instructions
    assert msgs[1] == history[0]
    assert msgs[2] == human_msg

def test_build_messages_list_with_str(client):
    msg = "hello as string"
    history = [AIMessage("prev")]
    msgs = client._build_messages_list(msg, history)
    assert msgs[0] == client.system_instructions
    assert msgs[1] == history[0]
    assert isinstance(msgs[2], HumanMessage)
    assert msgs[2].content == msg

def test_update_history(client):
    # Test with HumanMessage
    history = []
    human_msg = HumanMessage("hello from human")
    ai_msg = AIMessage("hello from ai")
    client._update_history(history, human_msg, ai_msg)
    assert history == [human_msg, ai_msg]

    # Test with list of ToolMessages
    history = []
    tool_msgs = [
        ToolMessage(tool_call_id="id1", content="tool result 1"),
        ToolMessage(tool_call_id="id2", content="tool result 2"),
    ]
    ai_msg2 = AIMessage("ai response considering tool messages")
    client._update_history(history, tool_msgs, ai_msg2)
    assert history == tool_msgs + [ai_msg2]

    # Test with string input
    history = []
    str_input = "hello as string input"
    ai_msg3 = AIMessage("ai response to string input")
    client._update_history(history, str_input, ai_msg3)
    assert history == [HumanMessage(str_input), ai_msg3]

    # Test mixed scenario: existing history
    history = [human_msg, ai_msg]
    new_human_msg = HumanMessage("new human input")
    new_ai_msg = AIMessage("new ai response")
    client._update_history(history, new_human_msg, new_ai_msg)
    assert history == [human_msg, ai_msg, new_human_msg, new_ai_msg]

def test_invoke_and_usage_metadata(client):
    human_msg = HumanMessage("hi")
    history = []
    response = client.invoke(human_msg, history)
    assert isinstance(response, AIMessage)
    assert len(client.usage_metadatas) == 1
    assert history[-1] == response
    # Check aggregated usage metadata
    agg_meta = client.get_usage_metadata()
    assert agg_meta.input_tokens == 1
    assert agg_meta.output_tokens == 2
    assert agg_meta.total_tokens == 3

def test_invoke_and_usage_metadata_with_string_input(client):
    str_input = "hi"
    history = []
    response = client.invoke(str_input, history)
    assert isinstance(response, AIMessage)
    assert len(client.usage_metadatas) == 1
    assert history[-1] == response
    # Check aggregated usage metadata
    agg_meta = client.get_usage_metadata()
    assert agg_meta.input_tokens == 1
    assert agg_meta.output_tokens == 2
    assert agg_meta.total_tokens == 3

def test_batch_invocation(client):
    msgs = [HumanMessage("hi1"), HumanMessage("hi2")]
    responses = client.batch(msgs)
    assert all(isinstance(r, AIMessage) for r in responses)
    assert len(client.usage_metadatas) == 1
    assert client.usage_metadatas[0][1].input_tokens == 2
    assert client.usage_metadatas[0][1].output_tokens == 4
    assert client.usage_metadatas[0][1].total_tokens == 6

def test_get_usage_metadata_from_timestamp(client):
    human_msg = HumanMessage("hi")
    client.invoke(human_msg)
    t0 = time.time()
    client.invoke(human_msg)
    agg_meta = client.get_usage_metadata(from_timestamp=t0)
    assert agg_meta.input_tokens == 1
    assert agg_meta.output_tokens == 2
    assert agg_meta.total_tokens == 3

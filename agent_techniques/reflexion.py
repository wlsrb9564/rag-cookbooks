# %% [markdown]
# # **Reflexion**
#
# Reflexion is a novel framework that introduces verbal reinforcement learning for language agents,
# allowing them to learn from past experiences and improve decision-making through self-reflection.
# Unlike traditional RL which requires parameter updates, Reflexion enhances agent performance
# using natural language-based self-improvement without modifying model weights.
#
# ### Key Concepts of Reflexion:
# 1. **Verbal Reinforcement Learning**: Agents receive textual feedback via self-reflections.
# 2. **Episodic Memory**: Past self-reflections are stored to inform future decisions.
# 3. **Three-Component Framework**:
#    - **Actor**: The LLM agent that performs tasks and generates outputs.
#    - **Evaluator**: Assesses the agent's output and provides feedback.
#    - **Self-Reflection Model**: Generates verbal reinforcement signals.
#
# Paper: https://arxiv.org/pdf/2303.11366

# %% [markdown]
# ## **Initial Setup**

# %%
import os
from dotenv import load_dotenv
load_dotenv()

# %% [markdown]
# ## **LLM & Tools**

# %%
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

# %%
# Set up Tavily Web Search API
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_tool = TavilySearchResults(max_results=5)

# %% [markdown]
# ## **Actor: Pydantic Schemas & Responder**
#
# - `Reflection`: Captures what is missing and superfluous in an answer.
# - `AnswerQuestion`: Full answer + reflection + search queries for improvement.
# - `ResponderWithRetries`: Wraps a chain with retry logic on validation errors.

# %%
import json
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ValidationError
from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question. Provide an answer, reflection, and then follow up with search queries to improve the answer."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: dict):
        response = []
        messages = state["messages"]
        for attempt in range(3):
            response = self.runnable.invoke(
                {"messages": messages}, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                self.validator.invoke(response)
                return {"messages": response}
            except ValidationError as e:
                messages.extend(
                    [
                        response,
                        ToolMessage(
                            content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                            + json.dumps(self.validator.tools[0].model_json_schema())
                            + " Respond by fixing all validation errors.",
                            tool_call_id=response.tool_calls[0]["id"],
                        ),
                    ]
                )
                state["messages"] = messages

        return {"messages": response}

# %% [markdown]
# ## **First Responder Chain**
#
# Builds the initial answer chain using the actor prompt template.
# The model is bound with `AnswerQuestion` as a structured output tool.

# %%
import datetime

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "\n\n<system>Reflect on the user's original question and the"
            " actions taken thus far. Respond using the {function_name} function.</reminder>",
        ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)
initial_answer_chain = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer.",
    function_name=AnswerQuestion.__name__,
) | llm.bind_tools(tools=[AnswerQuestion])
validator = PydanticToolsParser(tools=[AnswerQuestion])

first_responder = ResponderWithRetries(
    runnable=initial_answer_chain, validator=validator
)

# %%
# test first_responder
example_question = "Why is reflection useful in AI?"
initial = first_responder.respond(
    {"messages": [HumanMessage(content=example_question)]}
)

# %% [markdown]
# ## **Revisor Chain**
#
# Extends `AnswerQuestion` with a `references` field to enforce citation.
# The revisor uses the same actor prompt template but with revision instructions.

# %%
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Provide an answer, reflection,

    cite your reflection with references, and finally
    add search queries to improve the answer."""

    references: list[str] = Field(
        description="Citations motivating your updated answer."
    )


revision_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions,
    function_name=ReviseAnswer.__name__,
) | llm.bind_tools(tools=[ReviseAnswer])
revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator)

# %%
# test revisor
revised = revisor.respond(
    {
        "messages": [
            HumanMessage(content=example_question),
            initial["messages"],
            ToolMessage(
                tool_call_id=initial["messages"].tool_calls[0]["id"],
                content=json.dumps(
                    tavily_tool.invoke(
                        {
                            "query": initial["messages"].tool_calls[0]["args"][
                                "search_queries"
                            ][0]
                        }
                    )
                ),
            ),
        ]
    }
)
revised["messages"]

# %% [markdown]
# ## **Tool Node**
#
# `run_queries` runs Tavily searches in parallel for all generated search queries.
# `ToolNode` maps both `AnswerQuestion` and `ReviseAnswer` tool calls to the same function.

# %%
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)

# %% [markdown]
# ## **Build Graph**
#
# Graph flow: draft → execute_tools → revise → (loop back or END)
# The agent iterates up to `MAX_ITERATIONS` times before stopping.

# %%
from typing import Literal, Annotated
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]


MAX_ITERATIONS = 5
builder = StateGraph(State)
builder.add_node("draft", first_responder.respond)
builder.add_node("execute_tools", tool_node)
builder.add_node("revise", revisor.respond)

builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def _get_num_iterations(state: list):
    i = 0
    for m in state[::-1]:
        if m.type not in {"tool", "ai"}:
            break
        i += 1
    return i


def event_loop(state: list):
    num_iterations = _get_num_iterations(state["messages"])
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
builder.add_edge(START, "draft")
graph = builder.compile()

# %% [markdown]
# ## **Run**

# %%
events = graph.stream(
    {"messages": [("user", "How should we handle the climate crisis?")]},
    stream_mode="values",
)
for i, step in enumerate(events):
    print(f"Step {i}")
    step["messages"][-1].pretty_print()

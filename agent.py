import os
from typing import TypedDict, Annotated, List, Union
from dotenv import load_dotenv

# LangChain / LangGraph Imports
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Import your forecast engine from the previous section
from forecast import run_forecast

load_dotenv()

SYSTEM_PROMPT = """
You are an autonomous agent responsible for protecting the company's finances.

YOUR OPERATING PROTOCOL:
1. ALWAYS start by quantifying the risk using the 'forecast_cashflow_tool'.
2. IF the forecast shows "INCREASING (RISK)" or "CRITICAL SPIKE":
   - You MUST immediately call the 'read_policy_tool'.
   - Search for "cost control" or "travel restrictions" in the policy.
   - Your final answer MUST recommend specific actions from the policy (e.g., "Suspend business class travel").
3. Never just report the numbers. You must provide a solution.
"""

# --- 1. DEFINE TOOLS ---

@tool
def forecast_cashflow_tool(dummy_arg: str = "none"):
    """
    Use this tool when the user asks about financial future, risk, 
    burn rate, or cash flow projections. 
    It returns a trend analysis and a path to a plot image.
    """
    # We call the function we built in Section 2
    return run_forecast()

@tool
def read_policy_tool(query: str):
    """
    Searches the policy document for specific sections.
    """
    md_path = os.path.join("data", "docs", "policy.md")
    
    # Ensure static folder exists for the UI to read from
    if not os.path.exists("static"):
        os.makedirs("static")

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            
        # ... (Your existing search logic from the previous step) ...
        # ... logic to split by headers and find keywords ...
        # (Assuming you stuck with the simple search or the full text version)
        
        # --- LOGIC RECAP FOR CONTEXT ---
        sections = full_text.split("## ")
        relevant_sections = []
        query_words = query.lower().split()
        for section in sections:
            if any(word in section.lower() for word in query_words if len(word) > 4):
                relevant_sections.append("## " + section)
        
        # Determine the result text
        if relevant_sections:
            result_text = "\n---\n".join(relevant_sections[:3])
        else:
            result_text = "No specific policy section found."

        # === THE FIX: Save the result for the UI ===
        with open("static/active_policy.txt", "w", encoding="utf-8") as f:
            f.write(result_text)
        # ==========================================

        return result_text

    except Exception as e:
        return f"Error reading policy: {e}"

# List of tools to bind to the LLM
tools = [forecast_cashflow_tool, read_policy_tool]

# --- 2. SETUP THE LLM ---

# We use Groq for speed (Llama-3-8b is great for tool use)
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# We "bind" the tools to the LLM. 
# This teaches Llama-3 that these functions exist and how to call them.
llm_with_tools = llm.bind_tools(tools)

# --- 3. DEFINE STATE ---

class AgentState(TypedDict):
    # 'add_messages' ensures we keep the whole conversation history
    messages: Annotated[List[BaseMessage], add_messages]

# --- 4. DEFINE NODES ---

def call_model(state: AgentState):
    """
    The 'Thinking' Node. 
    It takes the conversation history, sends it to the LLM, 
    and gets back a response (which might be a tool call).
    """

    messages = state['messages']

    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# LangGraph has a pre-built node for running tools! 
# We don't even need to write the logic to execute the python function.
tool_node = ToolNode(tools)

# --- 5. DEFINE LOGIC (THE ROUTER) ---

def should_continue(state: AgentState):
    """
    This function decides the next step.
    If the LLM decided to call a tool -> Go to 'tools'
    If the LLM just answered ("Hello") -> Go to END
    """
    last_message = state['messages'][-1]
    
    # If the LLM response has 'tool_calls', we must run them
    if last_message.tool_calls:
        return "continue"
    return "end"

# --- 6. BUILD THE GRAPH ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set Entry Point
workflow.set_entry_point("agent")

# Add Conditional Edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

# Add Normal Edge
# After tools run, always go back to agent to interpret the result
workflow.add_edge("tools", "agent")

# Compile
app = workflow.compile()

# --- 7. TEST IT (Optional: Run this file directly) ---
if __name__ == "__main__":
    print("Agent being run...\n")
    
    # Simulate a user question
    user_input = "Analyze our financial risk for the next quarter."
    
    # Run the graph
    inputs = {"messages": [HumanMessage(content=user_input)]}
    result = app.invoke(inputs)
    
    # Print the final answer
    print("\n--- FINAL ANSWER ---")
    print(result['messages'][-1].content)
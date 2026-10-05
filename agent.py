from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
import gurobipy as gp
from google.adk.sessions import InMemorySessionService


from rich import print
import asyncio
from dotenv import load_dotenv
import uuid 
from pathlib import Path
load_dotenv() 

session_service = InMemorySessionService()



MODEL_GEMINI = "gemini-2.5-flash-lite"
MODEL_GPT = "openai/gpt-4o-mini"
MODEL_GPT_NANO = "openai/gpt-4.1-nano"
BASE_DIR = Path.cwd()
syntax_file_name = "gurobi_example.md"

def get_gurobi_syntax():
    """
    Looks up doc describing Gurobi syntax and examples
    """
    
    base = Path(__file__).parent  # directory of this Python file
    file_path = base / syntax_file_name

    with file_path.open("r", encoding="utf-8") as f:
        return f.read()


retry_config = types.HttpRetryOptions(
    attempts=2,  # Maximum retry attempts
    exp_base=30,  # Delay multiplier
    initial_delay=30,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)



def code_executor_function(code):
    namespace = {}
    try:
        exec(code, namespace, namespace)
        return(f"Execution completed successfully.\n****\n Result:\n{namespace.get("result")}")
    except Exception as e:
        return f"Error during code execution: {e}"



# -------------------------------------------------------------------
# 1) Interpreter Agent
#    - Reads user query
#    - Produces a structured *text* description of:
#      parameters, decision variables, constraints, objective
# -------------------------------------------------------------------

interpreter_agent = LlmAgent(
    name="InterpreterAgent",
    model=LiteLlm(model=MODEL_GPT),
    instruction="""You are a supply chain optimization model designer.

1. Carefully read the user query.
2. Design a linear / mixed-integer optimization model suitable for the Gurobi Python API.

Your output MUST be a clearly structured text with the following sections:

3. If It is a followup question, make sure to incorporate the new requirements into the model specification.
You have to change the values of parameters, decision variables, constraints, objective accordingly.
No need to mention what has changed, just provide the full updated model specification.

[PROBLEM_SUMMARY]
- A short description of what we are optimizing.

[PARAMETERS]
- List all sets and numerical parameters/variables needed for the model.
- For each parameter, specify:
  - name
  - description
  - entire set of the values (in JSON-like notation) read from user provided csv files or data
  - You have to provide the entire data for the parameters, not just the names, and no shortcuts


[DECISION_VARIABLES]
- List each decision variable:
  - name
  - description
  - index sets (e.g., plants, warehouses)
  - domain (continuous / integer / binary).

[CONSTRAINTS]
- For each constraint:
  - name
  - short description
  - a Python-like math expression, e.g.
    sum(x[i,j] for j in warehouses) <= capacity[i]

[OBJECTIVE]
- State if this is a minimization or maximization problem.
- Provide the objective expression in Python-like form, e.g.
  sum(cost[i,j] * x[i,j] for i in plants for j in warehouses)

Be explicit and consistent with names so another agent can translate this into code.
""",
    output_key="model_spec",  # stored in session state
)

print("✅ interpreter_agent created.")

# -------------------------------------------------------------------
# 2) Coder Agent
#    - Takes model_spec
#    - Writes Gurobi Python code
#    - Executes it with python_execution_tool
# -------------------------------------------------------------------

optimizer_agent = LlmAgent(
    name="OptimizerAgent",
    model=LiteLlm(model=MODEL_GPT),
    instruction="""You are a senior Python + Gurobi engineer.

You are given a textual model specification:
{model_spec}

TASK:
1. Generate a COMPLETE Python script that uses gurobipy to implement this model , 
2. Read data read and build variables from the model_specification, do not attempt to read from csv file
3. use the `get_gurobi_syntax` tool for syntax and examples to guide your code generation. 
4. use the following code snippet as a reference for using gurobi:
5. You always must have a "result" variable in your code that captures the final optimization output as a dictionary with keys: (ObjectiveValue, decision, metrics)
5.1 metrcis should include key constraint slacks and utilizations.
    ```python
    # Example usage of gurobi
    import gurobipy as gp
    from gurobipy import GRB
    model = gp.Model("example")
    model.addVar(<parameters>)
    constrName= model.addConstr(<constraint_expression>)
    model.setObjective(<objective_expression>))
    model.optimize()
    result = {
        "ObjectiveValue": model.ObjVal,

        "decision": {
            v.VarName: v.X
            for v in model.getVars()
        },

        "metrics": {
        for each entity in each set of production, warehouses, etc.:
            "Capacity": <calculate capacity metrics>,
            "Remaining": <calculate remaining capacity>,
            "Utilization":<calculate utilization>
               
    }
    ```
3. You are forbidden to calculate anything by yourself. 
4. You are to just generate the python code as per instructions and You must use the code_executor_function to execute the Python code that calculates the final optimization. 
If there are any errors, refer to the error messages and fix them, and retry execution.
5. Respond with the output from the code execution in an understandable format for the next agent. Do not change or modify andy numbers in the output, just present it clearly.
""",
    tools=[get_gurobi_syntax, code_executor_function],
    output_key="execution_result",  # tool result (string) stored here
)

print("✅ optimizer_agent created.")

# -------------------------------------------------------------------
# 3) Result Formatter Agent
#    - Reads model_spec + execution_result + user_query
#    - Produces a clear user-facing explanation
# -------------------------------------------------------------------

result_formatter_agent = LlmAgent(
    name="ResultFormatterAgent",
    model=LiteLlm(model=MODEL_GPT_NANO),
    instruction="""You are a supply chain analytics assistant.


Model specification:
{model_spec}

Raw solver output (return value + stdout):
{execution_result}

TASK:
1. Briefly explain in simple language what problem was modeled and solved.

2. If a feasible/optimal solution exists:
   - report the objective value (e.g., total cost).
   - List the decisions (e.g., shipment quantities by route).
    => use markdown table while displaying numbers
    => List all key decisions/variables with their values
    => If the list is long, show only key values on table and summarize rest with key highlights only.
3. For key constraints, mention their status:
    - report slack/surplus for each key constraint.
4. If the model is infeasible or unbounded:
   - explain what that means
   - suggest which constraints or assumptions might be adjusted.
5. Present the answer in a concise, well-structured format:
   - short paragraphs
   - bullet points for key numbers/decisions
   - avoid showing raw Python unless necessary.

This message will be shown directly to the user, so make it clear and readable.


OUTPUT_FORMAT:
# Problem Summary
<brief description>

# Key Results
- Objective Value: <value>
- Key Decisions:
| Decision | Value |
|-------------------|-------|
| ...               | ...   |
- Contraints Status / metrics:
| Constraint        | Slack/Surplus |
Mention the slack/surplus for key constraints.
- Highlights:

""",
    output_key="final_answer",
)

print("✅ result_formatter_agent created.")

# -------------------------------------------------------------------
# 4) Root Sequential Agent: Interpreter -> Coder -> ResultFormatter
# -------------------------------------------------------------------

root_agent = SequentialAgent(
    name="SupplyChainOptimizationPipeline",
    sub_agents=[interpreter_agent, optimizer_agent, result_formatter_agent],
)

print("✅ Sequential Agent created.")
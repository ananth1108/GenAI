import os
import json
from math import pow
from openai import OpenAI

# ----------  OpenAI client & model config  ----------
client = OpenAI(api_key="")

# You can swap to "o4-mini" for stronger reasoning (higher cost)
MODEL = "gpt-5-mini"

# ----------  Domain tools (pure Python functions)  ----------

def calculate_sip_projection(
    monthly_investment: float,
    expected_annual_return_pct: float,
    years: int,
) -> dict:
    """
    Simple SIP (Systematic Investment Plan) projection.
    Assumes fixed return and monthly compounding.
    """
    if years <= 0:
        return {
            "error": "Years must be > 0",
        }

    r = (expected_annual_return_pct / 100.0) / 12.0  # monthly rate
    n = years * 12                                   # months

    if r == 0:
        future_value = monthly_investment * n
    else:
        future_value = monthly_investment * ((pow(1 + r, n) - 1) / r)

    total_invested = monthly_investment * n
    gain = future_value - total_invested

    return {
        "monthly_investment": monthly_investment,
        "expected_annual_return_pct": expected_annual_return_pct,
        "years": years,
        "total_invested": round(total_invested, 2),
        "projected_value": round(future_value, 2),
        "projected_gain": round(gain, 2),
    }


def plan_retirement_goal(
    current_age: int,
    retirement_age: int,
    monthly_expense_today: float,
    inflation_pct: float,
    expected_return_pct_during_accumulation: float,
    current_corpus: float = 0.0,
) -> dict:
    """
    Very simplified retirement planner using:
    - Inflation for expense growth
    - 4% rule for corpus sizing
    - SIP formula to back-calc required monthly investment

    All numbers are rough estimates, NOT advice.
    """
    years_to_retirement = retirement_age - current_age
    if years_to_retirement <= 0:
        return {"error": "Retirement age must be greater than current age."}

    # Step 1: Inflate expenses to retirement
    infl = inflation_pct / 100.0
    target_monthly_at_retirement = monthly_expense_today * pow(1 + infl, years_to_retirement)

    # Step 2: 4% rule for required corpus
    annual_exp_at_retirement = target_monthly_at_retirement * 12
    safe_withdrawal_rate = 0.04  # 4% rule
    required_corpus = annual_exp_at_retirement / safe_withdrawal_rate

    # Step 3: Figure out SIP needed to reach that corpus
    goal_amount = max(required_corpus - current_corpus, 0)
    r = (expected_return_pct_during_accumulation / 100.0) / 12.0
    n = years_to_retirement * 12

    if n <= 0 or r <= 0:
        return {
            "error": "Invalid inputs for SIP calculation.",
            "required_corpus": round(required_corpus, 2),
        }

    # Reverse SIP formula:
    # FV = SIP * [((1+r)^n - 1) / r]  => SIP = FV * r / ((1+r)^n - 1)
    denom = (pow(1 + r, n) - 1)
    if denom == 0:
        return {
            "error": "Mathematical error in SIP calculation.",
            "required_corpus": round(required_corpus, 2),
        }

    required_monthly_sip = goal_amount * r / denom

    return {
        "current_age": current_age,
        "retirement_age": retirement_age,
        "years_to_retirement": years_to_retirement,
        "monthly_expense_today": round(monthly_expense_today, 2),
        "inflation_pct": inflation_pct,
        "expected_return_pct_during_accumulation": expected_return_pct_during_accumulation,
        "current_corpus": round(current_corpus, 2),
        "target_monthly_at_retirement": round(target_monthly_at_retirement, 2),
        "required_corpus": round(required_corpus, 2),
        "required_monthly_sip": round(required_monthly_sip, 2),
    }


# Map tool name -> Python function
TOOL_MAPPING = {
    "calculate_sip_projection": calculate_sip_projection,
    "plan_retirement_goal": plan_retirement_goal,
}

# ----------  Tool schemas for the model  ----------
# This follows the Responses API function-calling format. :contentReference[oaicite:1]{index=1}
TOOLS = [
    {
        "type": "function",
        "name": "calculate_sip_projection",
        "description": (
            "Calculate projected future value of a monthly investment (SIP) "
            "given years and expected annual return."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "monthly_investment": {
                    "type": "number",
                    "description": "Amount invested every month, in the user's currency.",
                },
                "expected_annual_return_pct": {
                    "type": "number",
                    "description": "Expected annual return percentage, e.g. 12 for 12%.",
                },
                "years": {
                    "type": "integer",
                    "description": "Number of years for investing.",
                    "minimum": 1,
                },
            },
            "required": [
                "monthly_investment",
                "expected_annual_return_pct",
                "years",
            ],
        },
    },
    {
        "type": "function",
        "name": "plan_retirement_goal",
        "description": (
            "Plan a simplified retirement goal. Use this when the user talks about "
            "retirement corpus, age, or long-term financial independence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "current_age": {
                    "type": "integer",
                    "description": "Current age of the user in years.",
                },
                "retirement_age": {
                    "type": "integer",
                    "description": "Desired retirement age in years.",
                },
                "monthly_expense_today": {
                    "type": "number",
                    "description": "Current monthly living expenses in today's money.",
                },
                "inflation_pct": {
                    "type": "number",
                    "description": "Assumed annual inflation percentage, e.g. 6 for 6%.",
                },
                "expected_return_pct_during_accumulation": {
                    "type": "number",
                    "description": (
                        "Expected annual return percentage on investments until retirement."
                    ),
                },
                "current_corpus": {
                    "type": "number",
                    "description": "Current investment corpus saved for retirement.",
                    "default": 0.0,
                },
            },
            "required": [
                "current_age",
                "retirement_age",
                "monthly_expense_today",
                "inflation_pct",
                "expected_return_pct_during_accumulation",
            ],
        },
    },
]

# ----------  Agentic loop helpers  ----------

def invoke_tools_from_response(response):
    """
    Look at the model's response and execute any function tools it requested.

    We return a list of `function_call_output` items to feed back into the model.
    This pattern is based on the official reasoning/function-calling cookbook. :contentReference[oaicite:2]{index=2}
    """
    messages_to_model = []

    for item in response.output:
        # We ignore "reasoning" items – they are internal explanations
        if item.type == "function_call":
            tool_name = item.name
            tool = TOOL_MAPPING.get(tool_name)
            if not tool:
                # If unknown tool name, send back an error for the model to handle
                tool_output = {
                    "error": f"No local implementation for tool '{tool_name}'"
                }
            else:
                try:
                    args = json.loads(item.arguments) if item.arguments else {}
                    tool_result = tool(**args)
                    tool_output = tool_result
                except Exception as e:
                    tool_output = {
                        "error": f"Exception while running tool '{tool_name}': {e}"
                    }

            messages_to_model.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(tool_output), # Fix: Convert tool_output to JSON string
                }
            )

    return messages_to_model


def run_agent_turn(user_input: str, previous_response_id: str | None = None):
    """
    One 'turn' of the agent:
    - Send user input
    - Let the model decide whether to call tools
    - If it calls tools, run them and loop until a final natural-language answer is produced.
    """
    # Initial call with the user's message
    response = client.responses.create(
        model=MODEL,
        input=user_input,
        tools=TOOLS,
        previous_response_id=previous_response_id,
    )

    while True:
        # If the model already gave us a final answer, output_text will be present
        function_messages = invoke_tools_from_response(response)

        if not function_messages:
            # No more tools to call => final assistant answer
            return response

        # Otherwise, feed tool outputs back so the model can continue reasoning
        response = client.responses.create(
            model=MODEL,
            input=function_messages,
            tools=TOOLS,
            previous_response_id=response.id,
        )


# ----------  Simple CLI loop  ----------

def main():
    print("=" * 70)
    print(" Simple Agentic AI Financial Planner (Demo)")
    print("  - Uses OpenAI Responses API + tool calling")
    print("  - Not investment advice; educational only")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 70)

    previous_response_id = None

    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Agent: Bye! Remember, always double-check with a human advisor.")
            break

        response = run_agent_turn(user, previous_response_id=previous_response_id)

        # Print the assistant's final answer text
        print("\nAgent:")
        # output_text is a convenience that combines message outputs :contentReference[oaicite:3]{index=3}
        print(response.output_text)

        previous_response_id = response.id


if __name__ == "__main__":
    main()

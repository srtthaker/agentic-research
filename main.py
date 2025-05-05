from smolagents import CodeAgent, TransformersModel, DuckDuckGoSearchTool
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('HUGGINGFACE_API_KEY')
login(token=api_key)
model_id = "meta-llama/Llama-3.1-8B-Instruct"

web_model = TransformersModel(
    model_id=model_id,
    max_new_tokens=4096,
    device_map="auto"
)

print("Model initialized. Creating agent...")

web_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()], 
    model=web_model,
    name="web_agent",
    description="Runs web searches for you."
)

print("Agent created. Running task...")

task = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"
web_result = web_agent.run(task)

print("\n--- Web Agent Result ---")
print(web_result)
print("--------------------")

def check_final_answer(answer, agent_memory):
    answer
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}"
        "Please check that the reasoning process and answer are correct: do they correctly answer the given task?"
        "Any run that invents numbers should fail."
    )

manager_agent = CodeAgent(
    model=model_id,
    tools=[],
    managed_agents=[web_agent],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_final_answer],
    max_steps=10
)

manager_result = manager_agent.run()
print("\n--- Manager Result ---")
print(manager_result)
print("--------------------")
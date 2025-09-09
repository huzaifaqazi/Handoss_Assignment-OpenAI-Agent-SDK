from agents import Agent, GuardrailFunctionOutput, RunConfig, RunContextWrapper, Runner, TResponseInputItem, input_guardrail , set_tracing_disabled , enable_verbose_stdout_logging , InputGuardrailTripwireTriggered , OutputGuardrailTripwireTriggered
from pydantic import BaseModel
from configuration.config import run_config , model
from handoff_agents.restaurant import restaurant_agent , guardrail_of_restaurant
from handoff_agents.train import  railway_agent , guardrail_of_train
from handoff_agents.school import school_agent , guardrail_of_school
set_tracing_disabled(True)

# Input Guardrails 
class Guardrail_data(BaseModel):
    is_political_question : bool
    reasoning:str

guardrail_agent = Agent(
    name="Gaurdrail Agent",
    instructions="check if user ask about politics questions.",
    output_type=Guardrail_data
)
@input_guardrail
async def political_guardrail(ctx : RunContextWrapper , agent : Agent , input : str) -> GuardrailFunctionOutput:
    answer = await Runner.run(guardrail_agent , input , context=ctx.context , run_config=run_config)
    return  GuardrailFunctionOutput(
        output_info=answer.final_output,
        tripwire_triggered=answer.final_output.is_political_question
    )

# Triage Main Agent
triage_agent = Agent(
    name="Triage agent",
    instructions="you are helpfull assistant.. and delegate task to other agents",
    handoffs=[restaurant_agent , school_agent , railway_agent],
    # output_guardrails=[guardrail_of_school , guardrail_of_restaurant , guardrail_of_train]
)

start_agent = triage_agent
input_data : list[TResponseInputItem] =[]
while True:
    prompt = input("Enter You Questions: ")
    if prompt == "end":
        break
    input_data.append({
        "role": "user",
        "content":prompt
    })
    try:
        res = Runner.run_sync(
        start_agent,
        input=input_data,
        run_config= RunConfig(model=model ,input_guardrails=[political_guardrail])
        )
        start_agent = res.last_agent
        input_data=res.to_input_list()

        print(f"Input Data History: {input_data}\n")
        print(res.final_output)
        print(res.last_agent.name)

    except InputGuardrailTripwireTriggered as e:
        print("input invalid triggered ⚠️" ,e)
    
    except OutputGuardrailTripwireTriggered as e:
        print("invalid output" , e)
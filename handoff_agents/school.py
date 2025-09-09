from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, Runner, output_guardrail
from pydantic import BaseModel
from handoff_agents.restaurant import restaurant_agent
from handoff_agents.train import railway_agent
from configuration.config import run_config

class output_of_school(BaseModel):
    response:str

class data_of_school(BaseModel):
    is_school_question: bool
    reason:str

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="check if llm output about school ",
    output_type=data_of_school
)

@output_guardrail
async def guardrail_of_school(ctx : RunContextWrapper , agent : Agent, output : output_of_school) -> GuardrailFunctionOutput:
    answer = await Runner.run(guardrail_agent , output.response , context=ctx , run_config=run_config)
    print("school question:",answer.final_output.is_school_question)
    print("Reason:" ,answer.final_output.reason)
    print("LLM Output : ",output.response)
    return GuardrailFunctionOutput(
        output_info=answer.final_output,
        tripwire_triggered= not answer.final_output.is_school_question 
    )

school_agent = Agent(
    name="School Agent",
    instructions="""you are School agent 
    The name of school is Metropolitan Academy School located near Ayesha Manzil. Admissions are open till 12 september.
    we offer from montessori to matric level also O'Level.
    fees pr student is 2500. 
    """,
    # handoffs=[restaurant_agent , railway_agent],
    handoff_description="also provide information about restaurant_agent and railway_agent ",
    output_type=output_of_school,
    output_guardrails=[guardrail_of_school]
)
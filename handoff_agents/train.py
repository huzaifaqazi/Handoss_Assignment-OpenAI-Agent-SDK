from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, Runner , function_tool, output_guardrail
from pydantic import BaseModel
from configuration.config import run_config

@function_tool
def train_tool(from_city : str , to_city:str):
    return f"The Shahlimar Express is shehdule {from_city} to {to_city} on 5 sep 2025."

class output_of_train(BaseModel):
    response:str

class data_of_train(BaseModel):
    is_train_question: bool
    reason:str

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="check output about train questions",
    output_type=data_of_train
)

@output_guardrail
async def guardrail_of_train(ctx : RunContextWrapper , agent : Agent, output : output_of_train) -> GuardrailFunctionOutput:
    answer = await Runner.run(guardrail_agent , output.response , context=ctx , run_config=run_config)

    print("Train question:",answer.final_output.is_train_question)
    print("Reason:" ,answer.final_output.reason)
    print("Response : ",output.response)
    return GuardrailFunctionOutput(
        output_info=answer.final_output,
        tripwire_triggered=answer.final_output.is_train_question is False
    )


railway_agent = Agent(
    name="Railway Agent",
    instructions="""You are a Railway agent. the name of Train is Shahlimar Express.
    Shahlimar Express have beautifull sitting area Economy ,A.C Stanadard and Bussiness Class .per ticket is 1k
    use can also call tool if user asking train shedule.
    """,
    # tools=[train_tool],
    output_type=output_of_train,
    output_guardrails=[guardrail_of_train]
)
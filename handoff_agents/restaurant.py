from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, Runner , function_tool, output_guardrail
from pydantic import BaseModel
from configuration.config import run_config

class output_of_restaurant(BaseModel):
    response:str

class data_of_restaurant(BaseModel):
    is_restaurant_question: bool
    reason:str

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="check the output if llm answer about Restaurant answer",
    output_type=data_of_restaurant
)

@output_guardrail
async def guardrail_of_restaurant(ctx : RunContextWrapper , agent : Agent, output : output_of_restaurant) -> GuardrailFunctionOutput:
    answer = await Runner.run(guardrail_agent , output.response , context=ctx , run_config=run_config)
    print("Restaurant question:",answer.final_output.is_restaurant_question)
    print("Reason:" ,answer.final_output.reason)
    print("LLM Output : ",output.response)
    return GuardrailFunctionOutput(
        output_info=answer.final_output,
        tripwire_triggered=answer.final_output.is_restaurant_question is False
    )


restaurant_agent = Agent(
    name="Restaurant Agent",
    instructions="""You are restuarant assistant. 
    1.The name of Restaurant is Ghousia Food & Grill Charga.,
    location of Restaurant in Karachi near Liaquatabad with beatifull sitting area for family. open for 24 hours but sunday is closed.
    2.Monal Restaurant in lahore location of Restaurant in near Badshashi Mosque with beatifull sitting area for family.Open for all days special discount for students.
    """,
    output_type=output_of_restaurant,
    output_guardrails=[guardrail_of_restaurant]
)
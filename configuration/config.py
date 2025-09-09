from agents import Agent , AsyncOpenAI , OpenAIResponsesModel , RunConfig , OpenAIChatCompletionsModel , set_tracing_export_api_key
from decouple import config


# api = config("openai_api_key")
# set_tracing_export_api_key(api)

gemini = config("gemini_api_key")
# base_url = config("base_url")
base_url = config("gemini_url")
model = config("model")

client = AsyncOpenAI(
    base_url=base_url,
    api_key=gemini
)

model = OpenAIChatCompletionsModel(
    openai_client=client,
    model=model
)
run_config = RunConfig(
    model=model,
    model_provider=client
)
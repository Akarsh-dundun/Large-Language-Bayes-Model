
import llb

text = (
    "I've been recording if it rains each day, with a 1 for rain and 0 for no rain. "
    "Maybe there's some kind of pattern? Predict if it will rain the next day."
)

rain = [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
data = {
    "num_days": len(rain),
    "rain": rain,
}

targets = ["outcome_for_next_day"]

API_KEY = "paste_your_api_key_here"
API_MODEL = "gpt-4.1-mini"

if API_KEY == "paste_your_api_key_here" or not API_KEY.strip():
    raise ValueError("Set API_KEY to a valid OpenAI key before running this script.")

posterior = llb.infer(
    text,
    data,
    targets,
    api_url="https://api.openai.com/v1/chat/completions",
    api_key=API_KEY,
    api_model=API_MODEL,
    n_models=16,
    mcmc_num_warmup=50,
    mcmc_num_samples=100,
)

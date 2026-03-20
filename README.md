# Large-Language-Bayes-Model

Large-Language-Bayes-Model is a lightweight pipeline for Bayesian inference with LLM-generated NumPyro models.

The project does the following:
- Takes a natural-language problem description and user data.
- Prompts an LLM to generate candidate NumPyro models.
- Runs MCMC inference for each valid model.
- Scores models with an approximate log marginal likelihood.
- Builds a weighted posterior by resampling from model posteriors.

## Example

```python
import llb

text = "I have a bunch of coin flips. What is the true bias of the coin?"
data = {"flips": [0, 0, 1, 0, 1, 0, 0]}
targets = ["true"]

posterior_samples = llb.infer(
	text=text,
	data=data,
	targets=targets,
	api_url="https://api.openai.com/v1/responses",
	api_key="YOUR_API_KEY",
	api_model="gpt-4.1-mini",
)
```

Expected output shape:

```python
{"true": [0.61, 0.41, 0.58, ...]}
```

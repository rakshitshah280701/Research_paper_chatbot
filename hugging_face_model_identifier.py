from huggingface_hub import HfApi

# Replace 'YOUR_HF_TOKEN' with your actual Hugging Face token
hf_token = "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"

# Initialize the Hugging Face API
api = HfApi()

# List all models you have access to
models = api.list_models(token=hf_token)

# Print the list of models
for model in models:
    print(model.modelId)

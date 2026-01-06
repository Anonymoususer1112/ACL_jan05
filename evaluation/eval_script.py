from model_interface import load_model
from dialect_eval import DialectEvaluator

# Load a model with the interface
model = load_model("gpt2")

# Create the evaluator
evaluator = DialectEvaluator(model=model, output_dir="results")

# Run evaluation 
results = evaluator.evaluate(
    data_file="path/to/dialect_pairs.csv",
    sae_column="standard_english", 
    aae_column="aae",
    sample_size=100,  # Set to None to use all examples
    batch_size=16
)

# Results will be saved to the output directory and returned as a dictionary
#!/bin/tcsh

# Script to calculate and analyze perplexity scores for dialectic bias research

# Set variables
set MODEL_NAME = "microsoft/phi-4"
set INPUT_FILE = "output_datasets/complete_aae_to_sae.csv"
set OUTPUT_FILE = "output_datasets/dialect_perplexity_results.csv"
set ANALYSIS_DIR = "results/dialect_perplexity_analysis"
set LIMIT = ""  # Empty means process all examples, use a number for testing

# Create output directories if they don't exist
mkdir -p ../output_datasets
mkdir -p $ANALYSIS_DIR

echo "Running dialect perplexity calculation..."

# Optional: If you want to test with a limited number of examples, uncomment this line
# set LIMIT = "--limit 100"

# Run the calculation script
python3 bias_analysis_scripts/calculate_dialect_perplexity.py \
    --model_name $MODEL_NAME \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    $LIMIT

# Check if the calculation was successful
if ($status == 0) then
    echo "Perplexity calculation completed successfully!"
    echo "Running analysis on results..."
    
    # Run the analysis script
    python3 analyze_perplexity_results.py \
        --input_file $OUTPUT_FILE \
        --output_dir $ANALYSIS_DIR
    
    if ($status == 0) then
        echo "Analysis completed successfully!"
        echo "Results are saved in ${OUTPUT_FILE}"
        echo "Visualizations are saved in ${ANALYSIS_DIR}"
    else
        echo "Error during analysis. Please check the log files."
    endif
else
    echo "Error during perplexity calculation. Please check the log files."
endif

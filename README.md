# Dialectic Preference Bias in Large Language Models

## Project Overview

This research investigates Dialectic Preference Bias in various Large Language Models. We define Dialectic Preference Bias as the phenomenon when LLMs output reflects or promotes unfair preferences or prejudices towards particular dialects or linguistic variations of a language. Such bias may lead the model to favor certain ways of speaking or writing, which can disadvantage speakers of marginalized dialects and perpetuate social biases and inequalities.

## Methodology

Our analysis uses a comparative approach across different language models:

1. We start with a dataset of tweets in African American English (AAE)
2. Translate the AAE tweets to Standard American English (SAE)
3. Translate the SAE tweets back to AAE
4. Obtain sentiment classifications for all three versions
5. Calculate bias metrics to assess dialect preference bias

## Key Findings

### Dialectic Group Invariance (DGI) Metrics

The DGI metric quantifies consistency in sentiment classification across dialects. Higher values indicate less bias.

*Running `bias_analysis_scripts/compare_models.py` will populate results here*

### Sentiment Shift Patterns

This table shows the percentage of samples where sentiment changed between dialects and the direction of bias.

*Running `bias_analysis_scripts/compare_models.py` will populate shifts results here*

* **Positive Bias**: Higher values indicate a tendency to assign more positive sentiment to SAE compared to AAE
* **Negative Bias**: Higher values indicate a tendency to assign more negative sentiment to SAE compared to AAE

## Visualizations

See the `/bias_analysis_scripts/results/visualizations` directory for detailed visualizations of the results.

## Running the Analysis

### Activate the virtual environment
`source dialectic/bin/activate`

### Make sure you have required dependencies
`pip install -r requirements.txt`

To run the bias analysis on your own data:

```
# Step 1: Translate AAE to SAE
python create_dataset_scripts/translate_aae_to_sae.py \
  --input source-dataset/cleaned_aae_dataset.csv \
  --output output_datasets/aae_to_sae_translations.csv

# Step 2: Translate SAE back to AAE
python create_dataset_scripts/translate_sae_to_aae.py \
  --input output_datasets/aae_to_sae_translations.csv \
  --output output_datasets/sae_to_aae_translations.csv

# Step 3: Get sentiments for each dialect version using each model
# (Repeat for each model: gpt4o_mini, claude_haiku, phi3_medium)
python obtain_sentiment_scripts/get_aae_sentiment.py \
  --input source-dataset/cleaned_aae_dataset.csv \
  --output output_datasets/MODEL_NAME_AAE_sentiment.csv \
  --model MODEL_NAME

# Step 4: Analyze bias for each model
python bias_analysis_scripts/analyze_dialectic_bias.py \
  --aae output_datasets/MODEL_NAME_AAE_sentiment.csv \
  --sae output_datasets/MODEL_NAME_SAE_sentiment.csv \
  --aae-from-sae output_datasets/MODEL_NAME_AAE_from_SAE_sentiment.csv \
  --output bias_analysis_scripts/results/MODEL_NAME_results.json \
  --model MODEL_NAME \
  --visualize

# Step 5: Compare models and update README
python bias_analysis_scripts/compare_models.py \
  --results-dir bias_analysis_scripts/results \
  --output bias_analysis_scripts/results/comparative_analysis \
  --visualize \
  --update-readme
```

## Conclusion

Our research reveals varying degrees of dialectic preference bias across the evaluated LLMs. The DGI metrics and sentiment shift patterns provide quantitative evidence of this bias, showing that these models do not treat different dialects equally when performing sentiment analysis tasks.

These findings highlight the importance of continued research and improvement in making language models more equitable across different linguistic varieties and dialects.

### Notes
- First is to finalize dataset.
- For the whole dataset, reproduce the AAAI paper output.
- Try performing these experiments:
  - For example, what words specifically cause higher bad sentiment in AAE?
  - Analysis of what tweets fall to neutral and negative when converted to AAE ( and if the reverse also happens when translated to AAE).
  - A good quantitative and qualitative analysis.
  - Topic modelling on the tweets and then analyzing each topic/cluster.
  - If possible, write a social impact analysis highlighting the impact on society (one crisp paragraph).
- Looking at other paper that are close to our benchmark idea.

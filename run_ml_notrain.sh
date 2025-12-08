cd /Users/bryce/Documents/code/h-lcdm

# Clear stages 3–5 checkpoints to refresh outputs
rm -f results/json/stage2_domain_adaptation_results.json \
      results/json/stage2_domain_adaptation.pkl \
      results/json/stage3_pattern_detection_results.json \
      results/json/stage4_interpretability_results.json \
      results/json/stage5_validation_results.json \
      results/json/stage5_validation.pkl \
      results/json/ml_results.json \ 
# Re-run pattern detection, interpretability, validation
python3 main.py --ml 
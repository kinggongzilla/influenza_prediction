#!/bin/bash

# Run inference for both Chronos and TiRex on all extracted data files
# Run once without test_split and once with test_split=0.1

# Create results directory if it doesn't exist
mkdir -p results

# List of data files to process (excluding example_data.csv)
data_files=(
    "data/argentina_ili_extracted.csv"
    "data/austria_ili_extracted.csv"
    "data/france_ili_extracted.csv"
    "data/france_ili_regular.csv"
    "data/germany_ili_extracted.csv"
    "data/italy_ili_extracted.csv"
    "data/mexico_ili_extracted.csv"
    "data/mongolia_ili_extracted.csv"
    "data/usa_ili_extracted.csv"
)

echo "Starting inference runs..."
echo "Total files to process: ${#data_files[@]}"
echo "Models: Chronos, TiRex"
echo "Modes: Normal, Evaluation (test_split=0.1)"
echo ""

# Function to run a single inference
run_inference() {
    local model=$1
    local data_file=$2
    local test_split=$3
    local output_prefix=$4
    
    echo "Running $model on $data_file (test_split=$test_split)..."
    
    if [ "$model" == "chronos" ]; then
        python src/scripts/chronos_inference.py \
            --data_file "$data_file" \
            --prediction_length 52 \
            ${test_split:+--test_split $test_split} \
            --plot
    elif [ "$model" == "tirex" ]; then
        python src/scripts/tirex_inference.py \
            --data_file "$data_file" \
            --prediction_length 52 \
            ${test_split:+--test_split $test_split} \
            --plot
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ $model completed successfully for $data_file"
    else
        echo "✗ $model failed for $data_file"
    fi
    echo ""
}

# Run inference for all files without test_split (normal mode)
echo "=== NORMAL MODE (no test_split) ==="
for data_file in "${data_files[@]}"; do
    run_inference "chronos" "$data_file" "" "normal"
    run_inference "tirex" "$data_file" "" "normal"
done

# Run inference for all files with test_split=0.1 (evaluation mode)
echo "=== EVALUATION MODE (test_split=0.1) ==="
for data_file in "${data_files[@]}"; do
    run_inference "chronos" "$data_file" "0.1" "eval"
    run_inference "tirex" "$data_file" "" "eval"
done

echo "All inference runs completed!"
echo "Results saved in the 'results/' directory"

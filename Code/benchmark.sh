#!/bin/bash

# Colors for output
NC='\033[0m' # No Color
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'

# Check if a file is provided as an argument
if [ -z "$1" ]; then
    echo -e "${RED}Error: No script file provided. Usage: ./benchmark.sh <filename>${NC}"
    exit 1
fi

# Store the script file name
file=$1

# Check if the file exists
if [ ! -f "$file" ]; then
    echo -e "${RED}Error: File '$file' not found!${NC}"
    exit 1
fi

# Run the script 10 times and capture the time for each run
times=()
echo -e "${BLUE}Run\tTime (seconds)${NC}" # Print the header

for i in {1..10}; do
    start=$(date +%s.%N)  # Capture start time
    python3 "$file" > /dev/null 2>&1  # Execute the Python script and suppress output
    end=$(date +%s.%N)    # Capture end time

    # Calculate the execution time in seconds
    real_time=$(echo "$end - $start" | bc)
    
    # Add the time to the array
    times+=($real_time)
    
    # Print the run number and time
    printf "${GREEN}%d\t%.2f${NC}\n" "$i" "$real_time"
done

# Calculate the average time
total=0
for t in "${times[@]}"; do
    total=$(echo "$total + $t" | bc)
done

average=$(echo "$total / 10" | bc -l)

# Print the average time
printf "\n${RED}Average time: %.2f seconds${NC}\n" "$average"

#!/bin/bash
set -euo pipefail

mkdir ./real_reports
mkdir ./sim_reports

find . -type f -name '*real_report.txt' -exec cp {} ./real_reports/ \;
find . -type f -name '*sim_report.txt' -exec cp {} ./sim_reports/ \;

script_dir="$(dirname -- "$(realpath "$0")")"
folder_name="$(basename -- "$script_dir")"

python3 "$script_dir/process_reports.py" \
        "$script_dir/real_reports/" \
        --output_csv "$script_dir/real_reports/${folder_name}_real_reports.csv"

python3 "$script_dir/process_reports.py" \
        "$script_dir/sim_reports/" \
        --output_csv "$script_dir/sim_reports/${folder_name}_sim_reports.csv"

cp -- "$script_dir/real_reports/${folder_name}_real_reports.csv" /dockerVLLM/raw_eval-data/new-analysis/eval-set/
cp -- "$script_dir/sim_reports/${folder_name}_sim_reports.csv" /dockerVLLM/raw_eval-data/new-analysis/eval-set/
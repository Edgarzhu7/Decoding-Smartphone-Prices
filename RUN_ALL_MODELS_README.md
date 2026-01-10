# Run All Models and Generate Report

## Overview

The `run_all_models_and_generate_report.py` script automatically runs all 14 models (7 types × 2 time dimensions) and generates a comprehensive PDF report summarizing the results.

## Usage

### Basic Usage

```bash
cd "/Users/dejiazhu/Desktop/ECON 495"
source env/bin/activate
python run_all_models_and_generate_report.py
```

### What It Does

1. **Runs all 14 models**:
   - 7 Quarterly models
   - 7 Annual models

2. **Extracts cumulative results** from Excel output files

3. **Generates PDF report** (`Model_Results_Summary_YYYYMMDD_HHMMSS.pdf`) containing:
   - Title page
   - Quarterly models summary table
   - Annual models summary table
   - Comparison charts (bar plots)
   - Model descriptions

## Models Run

### Quarterly Models
1. Traditional Jevons Index
2. Basic Lasso (Hedonic)
3. Basic Hedonic
4. Basic Hedonic + Error Feature
5. Delta Model
6. Delta Model + Error Feature
7. Time Dummy Model

### Annual Models
1. Traditional Jevons Index
2. Basic Lasso (Hedonic)
3. Basic Hedonic
4. Basic Hedonic + Error Feature
5. Delta Model
6. Delta Model + Error Feature
7. Time Dummy Model

## Output

### Console Output
- Progress for each model
- Success/failure status
- Execution summary

### PDF Report
The generated PDF includes:
- **Title Page**: Project information and generation timestamp
- **Quarterly Summary Table**: All quarterly models with cumulative Jevons indices
- **Annual Summary Table**: All annual models with cumulative Jevons indices
- **Comparison Charts**: Visual comparison of Traditional vs Hedonic indices
- **Model Descriptions**: Brief explanation of each model type

### Excel Files
All individual model outputs remain in their respective folders:
- `Quarter/*.xlsx`
- `annual/*.xlsx`

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- openpyxl
- All model scripts must be present and executable

## Notes

- Each model has a 10-minute timeout
- The script will continue even if some models fail
- Failed models will be listed in the execution summary
- The PDF will only include results from successfully completed models

## Example Output

```
============================================================
Running All Models and Generating Comprehensive Report
============================================================
Start time: 2025-10-XX XX:XX:XX

============================================================
Running: Quarterly Traditional Jevons Index
============================================================
✓ Successfully completed: Quarterly Traditional Jevons Index

...

============================================================
Extracting Results...
============================================================

============================================================
Generating PDF Report...
============================================================
✓ PDF report generated: Model_Results_Summary_202510XX_XXXXXX.pdf

============================================================
Execution Summary
============================================================
Models run: 14/14
PDF report: Model_Results_Summary_202510XX_XXXXXX.pdf
End time: 2025-10-XX XX:XX:XX
```

## Troubleshooting

### Models Fail to Run
- Check that all required Python packages are installed
- Verify that `Dataset.xlsx` exists in the project root
- Check individual model scripts for errors

### PDF Generation Fails
- Ensure matplotlib is properly installed
- Check that Excel output files exist from successful model runs
- Verify write permissions in the project directory

### Missing Results in PDF
- Some models may have failed - check the execution summary
- Excel files may not have the expected sheet names
- Results extraction may need adjustment for different output formats



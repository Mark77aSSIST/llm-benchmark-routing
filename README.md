# LLM Benchmark Routing Experiments

Files and guides to reproduce the experiments of the paper "Cost-Effective Large Language Model Ensemble with Adaptive Benchmark-based Weighting and Dynamic Model Selection".

## 📝 Authors
Mukeun Choi_1 and Taeyeon Oh_2*

* 1 : Seoul AI School, aSSIST University, Seoul, Republic of Korea
* 2 : Seoul AI School, aSSIST University, Seoul, Republic of Korea
* *: Corresponding Author

## 📦 File Path

| Path | Description |
|------|------|
| `data` | Raw & Preprocessed Data |
| `results` | Evaluation Results |
| `scripts` | Experiment Code |
| `src` | Files generated during the experiment (json log and pkl) |

### 🔍 Experiment Code in scripts path

| File | Description |
|------|------|
| `01_Data_preprocessing.py` | Library import & Data Preprocessing |
| `02_Define_variablesAndFunctions.py` | Variables & Functions Definition |
| `03_Model_definitions.py` | Model Definition |
| `04_Evaluation.py` | Evaluation |
| `05_Visualization.py` | Figure & Table Generation |

### 📚 Reference File

| File | Description |
|------|------|
| `README.md` | This File |
| `requirements.txt` | Library installation information required to run in a local environment |

## 📊 Estimated Experimental Results

```
Method               Accuracy    Avg Calls
Oracle                100.00%        0.00
Best Single            46.71%        1.00
Uniform Ensemble       58.18%       11.00
ABW                    84.86%       11.00
SPW                    58.18%       11.00
Hybrid                 84.75%       11.00
Top-K                  85.02%        1.00

Top-K Distribution:
K=1: 6348 (100.0%)
K=2:    0 (  0.0%)
K=3:    0 (  0.0%)
```

## 🚀 Quick Start

1) Login to Google Colab environment (T4 GPU 16GB RAM)
2) Copy and paste Python files in the "scripts" folder into each cell in order of number
3) Upload raw data file to your Google Drive folder (ex. routerbench_datasets)
4) Connect Google Drive to your colab file
5) Set up and verify the path in the code
6) Run each cell's code in order
7) You can check and download the results

### Set up the file path (just example)

```python
DATA_DIR = "/content/drive/MyDrive/routerbench_datasets"
WORK_DIR = "/content/routerbench_results"
```

## ✅ Raw Data Source

Thank you to team RouterBench (Hu et al, 2024) for providing the data.
The RouterBench dataset (raw-data) used in this study is available at the following link.
* Dataset Main: https://github.com/withmartian/routerbench?tab=readme-ov-file
* Dataset Download: https://huggingface.co/datasets/withmartian/routerbench


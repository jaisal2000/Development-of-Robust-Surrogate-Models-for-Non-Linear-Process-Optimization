# Surrogate Modeling for Binary Distillation Optimization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Focus-Scientific%20ML-green)
![Status](https://img.shields.io/badge/Status-Research%20Complete-success)

## 📌 Project Overview
This project implements **Scientific Machine Learning (SciML)** techniques to develop robust surrogate models for a binary Ethanol-Water distillation column. By automating a physics-based simulator (**DWSIM**) via Python, we generated a dataset representing the **NRTL thermodynamic landscape**.

The core objective was not just accuracy, but **physical consistency**. We evaluated multiple regression architectures (Polynomial, Random Forest, Gradient Boosting) to determine which models adhere to process physics (monotonicity) and which fail during extrapolation into high-reflux operating regions.

## 🚀 Key Features
*   **Process Automation:** Automated DWSIM flowsheet execution to generate 446 convergent data points across a wide design space ($0.8 \le R \le 5.0$).
*   **Physics-Constrained Evaluation:** Implemented diagnostic checks to ensure model predictions respect physical bounds ($0 \le x_D \le 1$) and monotonic trends.
*   **Extrapolation Stress-Testing:** Investigated the mathematical divergence between **global continuous functions** (Polynomial Regression) and **local piecewise approximations** (Tree-based models) when predicting outside the training domain.

## 🛠️ Methodology
1.  **Data Generation:** 
    *   *Tool:* DWSIM (Open Source Process Simulator)
    *   *System:* Ethanol-Water (NRTL Property Package)
    *   *Variables:* Reflux Ratio, Boilup Ratio, Feed Composition, Feed Flow.
2.  **Modeling Strategy:**
    *   **Polynomial Regression (Degree 3):** Chosen for its superior generalization across data gaps.
    *   **Random Forest / Gradient Boosting:** High local accuracy ($R^2 > 0.99$) but failed catastrophically during extrapolation ($R^2 < 0$).

## 📂 File Description
*   `automation_flowsheet_simulation.py`: Handles the interface with DWSIM, parametrizes the flowsheet, and manages the simulation loops to generate `distillation_sim_results.csv`.
*   `data_processing_modeling.py`: Contains the pre-processing pipeline (StandardScaler), model training (Scikit-Learn), hyperparameter tuning (GridSearchCV), and the custom extrapolation analysis logic.

## 📊 Key Results
| Model | Interpolation (Validation) | Extrapolation (Test Set) | Physics Compliance |
| :--- | :---: | :---: | :---: |
| **Polynomial Reg.** | High Accuracy | **Robust** | High |
| **Random Forest** | **Near Perfect** | **Failed (Negative $R^2$)** | Low (outside bounds) |

*The study concludes that while tree-based models offer superior interpolation, global polynomial functions are required for safe optimization in undefined operating regions.*

## 📦 Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/distillation-surrogate-modeling.git

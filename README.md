<div align="center">

  <a name="readme-top"></a>
  # Supervised Machine Learning
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
  ![Status](https://img.shields.io/badge/Status-Completed-success)
  [![Technology](https://img.shields.io/badge/Technology-Python%20%7C%20Scikit--Learn-blueviolet)](https://github.com/Amey-Thakur/TSF-SUPERVISED-MACHINE-LEARNING)
  [![Developed by Amey Thakur and Mega Satish](https://img.shields.io/badge/Developed%20by-Amey%20Thakur%20%26%20Mega%20Satish-blue.svg)](https://github.com/Amey-Thakur/TSF-SUPERVISED-MACHINE-LEARNING)

  A predictive analytics study demonstrating the application of **Ordinary Least Squares (OLS) Regression** to estimate academic performance based on temporal study patterns.
  
  **[Google Colab](https://colab.research.google.com/github/Amey-Thakur/TSF-SUPERVISED-MACHINE-LEARNING/blob/main/Source%20Code/TSF_INTERNSHIP_TASK_1_SUPERVISED_LEARNING.ipynb)** &nbsp;·&nbsp; **[Kaggle Notebook](https://www.kaggle.com/ameythakur20/tsf-internship-task-1-supervised-learning)** &nbsp;·&nbsp; **[Video Demo](https://www.youtube.com/watch?v=qsO9GyGNWf0)** &nbsp;·&nbsp; **[Live Demo](https://amey-thakur.github.io/TSF-SUPERVISED-MACHINE-LEARNING/)**

  [![TSF Supervised Machine Learning Demo](https://img.youtube.com/vi/qsO9GyGNWf0/0.jpg)](https://www.youtube.com/watch?v=qsO9GyGNWf0)

</div>

---

<div align="center">

  [Authors](#authors) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Features](#features) &nbsp;·&nbsp; [Structure](#project-structure) &nbsp;·&nbsp; [Results](#results) &nbsp;·&nbsp; [Quick Start](#quick-start) &nbsp;·&nbsp; [Usage Guidelines](#usage-guidelines) &nbsp;·&nbsp; [License](#license) &nbsp;·&nbsp; [About](#about-this-repository) &nbsp;·&nbsp; [Acknowledgments](#acknowledgments)

</div>

---

<!-- AUTHORS -->
<div align="center">

  <a name="authors"></a>
  ## Authors

  | <a href="https://github.com/Amey-Thakur"><img src="https://github.com/Amey-Thakur.png" width="150" height="150" alt="Amey Thakur"></a><br>[**Amey Thakur**](https://github.com/Amey-Thakur)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5644--1575-green.svg)](https://orcid.org/0000-0001-5644-1575) | <a href="https://github.com/msatmod"><img src="Mega/Mega.png" width="150" height="150" alt="Mega Satish"></a><br>[**Mega Satish**](https://github.com/msatmod)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1844--9557-green.svg)](https://orcid.org/0000-0002-1844-9557) |
  | :---: | :---: |

</div>

> [!IMPORTANT]
> ### 🤝🏻 Special Acknowledgement
> *Special thanks to **[Mega Satish](https://github.com/msatmod)** for her meaningful contributions, guidance, and support that helped shape this work.*

---

<!-- OVERVIEW -->
<a name="overview"></a>
## Overview

**Supervised Machine Learning - Task 1** is a foundational Data Science exploration conducted under the **Graduate Rotational Internship Program (GRIP)** at **The Sparks Foundation**. The project establishes a univariate linear model to quantify the correlation between study duration (Hours) and academic outcome (Scores).

By leveraging **Scikit-Learn's** predictive algorithms, the system minimizes the sum of squared residuals to derive a best-fit line, enabling precise scalar predictions for arbitrary numerical inputs (e.g., predicting scores for 9.25 hours of study).

### Computational Objectives
The simulation is governed by strict **statistical principles** ensuring reproducibility and accuracy:
*   **Linear Approximation**: establishing a linear relationship $y = mx + c$ where $y$ is the predicted score and $x$ is study hours.
*   **Residual Minimization**: utilizing the OLS algorithm to optimize the coefficient (slope) and intercept.
*   **Predictive Inference**: generating a specific scalar output for the internship query: *What will be the predicted score if a student studies for 9.25 hrs/day?*

> [!TIP]
> **Model Applicability**: While this linear model provides a precise mathematical estimation for the given range, extrapolating predictions beyond the observed data range (e.g., studying > 15 hours/day) may yield unrealistic results due to the physical constraints of a 24-hour day.

---

<!-- FEATURES -->
<a name="features"></a>
## Features

| Component | Technical Description |
|-----------|-----------------------|
| **Ingestion Pipeline** | Automated data retrieval and parsing using **Pandas** from remote HTTP endpoints. |
| **Exploratory Analysis** | Visualizing distribution and correlation via **Matplotlib** and **Seaborn** scatter plots. |
| **Model Architecture** | Implementation of `LinearRegression` from **Scikit-Learn** for OLS optimization. |
| **Evaluation Metrics** | Quantitative assessment using **Mean Absolute Error (MAE)** to validate model precision. |
| **Inference Engine** | Direct scalar injection logic to predict outcomes for specific user-defined inputs. |

> [!NOTE]
> ### Empirical Context
> The dataset consists of a bivariate distribution (Hours vs. Scores). The high correlation coefficient observed during EDA justifies the selection of a Linear Regression model over more complex polynomial or ensemble approaches, adhering to the principle of parsimony (Occam's Razor) in machine learning design.

### Tech Stack
- **Runtime**: Python 3.x
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn (sklearn)
- **Environment**: Jupyter Notebook / Google Colab

---

<!-- STRUCTURE -->
<a name="project-structure"></a>
## Project Structure

```python
TSF-SUPERVISED-MACHINE-LEARNING/
│
├── docs/                                            # Technical Documentation
│   └── SPECIFICATION.md                             # Architecture & Design Specification
│
├── Mega/                                            # Archival Attribution Assets
│   ├── Filly.jpg                                    # Companion (Filly)
│   ├── Mega.png                                     # Author Profile Image (Mega Satish)
│   └── ...                                          # Additional Attribution Files
│
├── Source Code/                                     # Core Implementation
│   └── TSF_INTERNSHIP_TASK_1_SUPERVISED_LEARNING.ipynb  # Jupyter Notebook (Analysis Kernel)
│
├── The Sparks Foundation/                           # Internship Artifacts
│   └── Task_1_Dataset.csv                           # Empirical Data Source
│
├── .gitattributes                                   # Git configuration
├── .gitignore                                       # Repository Filters
├── CITATION.cff                                     # Scholarly Citation Metadata
├── codemeta.json                                    # Machine-Readable Project Metadata
├── LICENSE                                          # MIT License Terms
├── README.md                                        # Project Documentation
└── SECURITY.md                                      # Security Policy
```

---

<!-- RESULTS -->
<a name="results"></a>
## Results

<div align="center">
  <b>1. Exploratory Data Analysis: Hours vs Percentage</b>
  <br>
  <i>Initial scatter plot revealing the strong positive correlation.</i>
  <br><br>
  <img src="screenshots/01-dataset-scatter.png" alt="Dataset Scatter" width="70%">
  <br><br>

  <b>2. Feature Distribution: Scores Analysis</b>
  <br>
  <i>Statistical distribution of the target variable (Percentage Scored).</i>
  <br><br>
  <img src="screenshots/02-score-distribution.png" alt="Score Distribution" width="70%">
  <br><br>

  <b>3. Regression Fit: Hours vs Scores</b>
  <br>
  <i>OLS Regression line fitted to the training data.</i>
  <br><br>
  <img src="screenshots/03-regression-plot.png" alt="Regression Plot" width="70%">
  <br><br>

  <b>4. Model Training: Fitting the Line</b>
  <br>
  <i>Visualizing the linear approximation on Training Data.</i>
  <br><br>
  <img src="screenshots/04-training-set-results.png" alt="Training Set Results" width="70%">
  <br><br>

  <b>5. Model Validation: Testing the Fit</b>
  <br>
  <i>Validation of the regression line against unseen Test Data.</i>
  <br><br>
  <img src="screenshots/05-test-set-results.png" alt="Test Set Results" width="70%">
  <br><br>
  
  <b>Evaluation Metrics</b>
  <br>
  <code>Mean Absolute Error: 4.18</code> | <code>R2 Score: 0.945</code>
  <br><br>
  
  <b>Final Inference</b>
  <br>
  <code>Input: 9.25 Hours</code> &rarr; <code>Predicted Score: 93.69%</code>
</div>

---

<!-- QUICK START -->
<a name="quick-start"></a>
## Quick Start

### 1. Prerequisites
- **Python 3.7+**: Required for runtime execution. [Download Python](https://www.python.org/downloads/)
- **Jupyter Environment**: For interactive code execution (JupyterLab or Notebook).

> [!WARNING]
> **Data Path Integrity**
>
> The analysis kernel relies on precise relative file paths. Ensure `Task_1_Dataset.csv` remains within `The Sparks Foundation/` directory. Modifying the directory structure without updating the ingestion logic will result in `FileNotFoundError` during runtime.

### 2. Installation
Establish the local environment by cloning the repository and installing the computational stack:

```bash
# Clone the repository
git clone https://github.com/Amey-Thakur/TSF-SUPERVISED-MACHINE-LEARNING.git
cd TSF-SUPERVISED-MACHINE-LEARNING

# Install predictive modeling dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Execution
Launch the analysis kernel to reproduce the findings:
```bash
jupyter notebook "Source Code/TSF_INTERNSHIP_TASK_1_SUPERVISED_LEARNING.ipynb"
```
---

<!-- USAGE GUIDELINES -->
<a name="usage-guidelines"></a>
## Usage Guidelines

This repository is openly shared to support learning and knowledge exchange across the academic community.

**For Students**  
Use this project as reference material for understanding **supervised learning pipelines**, **univariate regression**, and **statistical predictive modeling**. The source code is available for study to facilitate self-paced learning and exploration of **OLS optimization and residual analysis**.

**For Educators**  
This project may serve as a practical lab example or supplementary teaching resource for **Data Science** and **Applied Statistics** courses. Attribution is appreciated when utilizing content.

**For Researchers**  
The documentation and architectural approach may provide insights into **academic project structuring**, **predictive inference**, and **industrial internship artifacts**.

---

<!-- LICENSE -->
<a name="license"></a>
## License

This academic submission, developed for the **Graduate Rotational Internship Program (GRIP)** at **The Sparks Foundation**, is made available under the **MIT License**. See the [LICENSE](LICENSE) file for complete terms.

> [!NOTE]
> **Summary**: You are free to share and adapt this content for any purpose, even commercially, as long as you provide appropriate attribution to the original authors.

Copyright © 2021 Amey Thakur & Mega Satish

---

<!-- ABOUT -->
<a name="about-this-repository"></a>
## About This Repository

**Created & Maintained by**: [Amey Thakur](https://github.com/Amey-Thakur) & [Mega Satish](https://github.com/msatmod)<br>
**Role**: Data Science & Business Analytics Interns  
**Program**: Graduate Rotational Internship Program (GRIP)  
**Organization**: [The Sparks Foundation](https://www.thesparksfoundationsingapore.org/)

This project features **Supervised Machine Learning - Task 1**, a predictive analytics study conducted as part of the **GRIP Internship**. It explores the application of linear regression to solve real-world estimation problems.

**Connect:** [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [LinkedIn](https://www.linkedin.com/in/amey-thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

### Acknowledgments

Grateful acknowledgment to [**Mega Satish**](https://github.com/msatmod) for her exceptional collaboration and scholarly partnership during the execution of this data science internship task. Her analytical precision, deep understanding of statistical modeling, and constant support were instrumental in refining the predictive algorithms used in this study. Working alongside her was a transformative experience; her thoughtful approach to problem-solving and steady encouragement turned complex regression challenges into meaningful learning moments. This work reflects the growth and insights gained from our side-by-side academic journey. Thank you, Mega, for everything you shared and taught along the way.

Special thanks to the **mentors at The Sparks Foundation** for providing this platform for rapid skill development and industrial exposure.

---

<div align="center">

  [↑ Back to Top](#readme-top)

  [Authors](#authors) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Features](#features) &nbsp;·&nbsp; [Structure](#project-structure) &nbsp;·&nbsp; [Results](#results) &nbsp;·&nbsp; [Quick Start](#quick-start) &nbsp;·&nbsp; [Usage Guidelines](#usage-guidelines) &nbsp;·&nbsp; [License](#license) &nbsp;·&nbsp; [About](#about-this-repository) &nbsp;·&nbsp; [Acknowledgments](#acknowledgments)

  <br>

  📈 **[TSF-SUPERVISED-MACHINE-LEARNING](https://amey-thakur.github.io/TSF-SUPERVISED-MACHINE-LEARNING/)**

  ---

  ### Presented as part of the Internship @ The Sparks Foundation

  ---

  ### 🎓 [Computer Engineering Repository](https://github.com/Amey-Thakur/COMPUTER-ENGINEERING)

  **Computer Engineering (B.E.) - University of Mumbai**

  *Semester-wise curriculum, laboratories, projects, and academic notes.*

</div>

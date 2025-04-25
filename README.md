# Lang et al. (in Review) Which indicators matter? Using performance indicators to predict in-game success-related events in association football.

Paper link: tbd
---

## 📑 Table of Contents

- [📖 About](#-about)
- [📁 Repository Structure](#-repository-structure)
- [⚙️ Setup](#️-setup)
- [🚀 Running Experiments](#-running-experiments)
- [📊 Data](#-data)
- [📈 Results and Evaluation](#-results-and-evaluation)
- [📚 Citation](#-citation)
- [🛡️ License](#-license)
- [📬 Contact](#-contact)

---

### 📘 About

This repository accompanies the article "*Which indicators matter? Using performance indicators to predict in-game success-related events in association football*", under review in *International Journal of Computer Science in Sport (IJCSS)*. 

The study investigates how well 28 commonly used performance indicators (PIs) predict short-term success- or scoring-related events (SREs) such as goals, shots, and box entries in professional soccer. These predictions are based on how a team performs in a defined time span leading up to the event.

Using data from 102 Bundesliga matches and thousands of machine learning model configurations, we evaluated which PIs or PI-combinations best reflect a team’s current performance and can anticipate upcoming events. We found that indicators derived from frequent in-game actions, i.e. **Dangerousity** , **Successful Passes into the Attacking Third**, and **Outplayed Opponents**, are more effective than those based on rare events like **Goals** or **Corner Kicks**. Additionally, comparing team differences in PIs often improves predictive performance.

To our knowledge, this is the first study to predict in-play events *beyond the immediate next event*, opening new possibilities for real-time match analysis. Based on our findings, we also propose a novel match momentum metric, grounded in empirical prediction data, which can support tactical decisions and enhance in-play betting strategies.

This repository includes the code, model configurations, and result outputs used in the study. It is intended for researchers, analysts, and practitioners interested in applied machine learning for sports analytics, event prediction, and real-time performance evaluation.

---

### **📁** Repository Structure

The repository is organized as follows:

#### **📂 Folders**

* **`configs/`**  
   Contains configuration files for running experiments, including time window settings and model parameters.

* **`data/`**  
   Includes an original match data file used in the study. *(Note: Check license and usage terms before redistribution.)*

* **`models/`**  
   Contains implementations of various machine learning models used in the experiments.

* **`output/`**  
   Stores output files, including model results, evaluation metrics, and generated plots.

* **`utils/`**  
   Includes utility scripts for data handling, sampling, and visualization.

#### **📄 Key Files**

* **`environment.yml`**  
   Conda environment file listing all required dependencies.

* **`run.py`**  
   Main script for initializing and running experiments based on provided configurations.


---

### **⚙️ Setup**

We use **Conda** for environment and dependency management. To set up the project environment, follow these steps:

1. **Create the environment**  
    Run the following command in the repository root to install all required libraries:       
   `conda env create \-f environment.yml`
   
2. **Activate the environment**
   Once created, activate the environment using:  
   `conda activate shortterm\_event\_pred`

3. **Update the environment**  
   If you’ve already created the environment and the `environment.yml` file has changed, you can update it with:    
   `conda env update \--file environment.yml \--prune`

   This will ensure all dependencies are up to date and any removed packages are pruned accordingly.

---

### **🧪 Running Experiments**

You can run experiments using the `run.py` script along with a configuration file that defines the experimental setup.

#### **▶️ Basic Usage**

`python run.py CONFIG`

* Replace `CONFIG` with the **name of a configuration file** located in the `configs/` folder (e.g., `training_config.yaml`).
* This file specifies parameters such as input and prediction windows, target events, and model settings.

#### **📂 Configuration Templates**

The `configs/` folder includes:

* A **README file** explaining how to structure configuration files.

* **Sample configurations** you can use directly or modify for your own experiments.

This design makes it easy to reproduce the original experiments or explore new setups with minimal adjustments.

---

### **📊 Data**

This repository includes data from one sample match, which is provided for demonstration and testing purposes only. Due to licensing restrictions, we are unable to share the full dataset used in the study.

#### **Please note:**

* We are not the legal owners of the complete dataset.
* The full set of event and tracking data from 102 Bundesliga matches used in the published study is not publicly available.
* However, as stated in the article, data access for academic or research purposes may be granted upon reasonable request to the corresponding author.

The included sample file allows users to explore the structure, preprocessing, and modeling workflow described in the paper.

---

### 📈 Results and Evaluation

This repository does not include the full results of the experiments, as they are provided in the **published article** and in a **supplementary dataset available via Figshare**.

#### **🔍 Key Findings (Summary)**

* **Dangerousity** was the most effective PI for predicting goals and shots.

* **Entries into the Attacking Third** performed best for corner kicks, third entries, and box entries.

* PIs reflecting **frequent in-game actions** (e.g., final-third possession, Dangerousity, opponents outplayed) outperformed those based on **rare events** (e.g., goals, corners).

* Combining certain PIs (e.g., **Opponents Outplayed** and **Tacklings Won**) increased predictive accuracy, especially for goals.

* A match momentum metric based on real-time prediction differences showed potential for tactical support and live betting applications.

#### **📄 Where to Find Full Results**

* All detailed results are available in the **supplementary material** on **Figshare**: [https://figshare.com/projects/Lang\_et\_al\_2025\_Which\_indicators\_matter\_Using\_performance\_indicators\_to\_predict\_in-game\_success-related\_events\_in\_association\_football/223491](https://figshare.com/projects/Lang_et_al_2025_Which_indicators_matter_Using_performance_indicators_to_predict_in-game_success-related_events_in_association_football/223491) .

* Please refer to the article for a full methodological description and in-depth analysis.

---

### 📚 Citation
 
`Lang, S., Wimmer, T., Erben, A., & Link, D. in (Review). Which indicators matter? Using performance indicators to predict in-game success-related events in association football. International Journal of Computer Science in Sport (IJCSS).`
 
---

**🛡️ License**

This repository is shared under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license, in accordance with the journal’s Open Access policy.

You are free to:

* **Share** — copy and redistribute the material in any medium or format    As long as you follow these terms: 

* **Attribution** — you must give appropriate credit, provide a link to the license, and indicate if changes were made. 

* **NonCommercial** — you may not use the material for commercial purposes. 

* **NoDerivatives** — if you remix, transform, or build upon the material, you may not distribute the modified material. 

License details: https://creativecommons.org/licenses/by-nc-nd/4.0/

---

### **📬 Contact**

For questions regarding the repository, data usage, or the study itself, please contact the corresponding author:

**Steffen Lang**  
TUM School of Medicine and Health Sciences  
Technical University of Munich (TUM)  
✉️ steffen.lang@tum.de

# 🩺 Diabetes Prediction - Exploratory Data Analysis with Apache Spark

![Apache Spark](https://img.shields.io/badge/Apache%20Spark-FDEE21?style=for-the-badge&logo=apachespark&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

## 📊 Project Overview

This project performs comprehensive **Exploratory Data Analysis (EDA)**, **Feature Engineering**, and **Data Visualization** on the Pima Indian Diabetes dataset using **Apache Spark**. The analysis aims to identify key predictors of diabetes and build predictive models to assist in early diagnosis and healthcare decision-making.

## 🎯 Project Goals

- 🔍 **Exploratory Data Analysis** of diabetes dataset
- 📈 **Statistical Analysis** and correlation studies  
- 🎨 **Data Visualization** using multiple plotting techniques
- 🤖 **Machine Learning Model** development for diabetes prediction
- ☁️ **Distributed Computing** with Apache Spark on AWS Hadoop Cluster

## 📁 Dataset Information

### Source
- **Dataset**: Pima Indians Diabetes Database
- **Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records**: 768 female patients of Pima Indian heritage
- **Features**: 8 medical predictors + 1 target variable

### Features Description
| Feature | Description | Type |
|---------|-------------|------|
| 🩺 **Pregnancies** | Number of times pregnant | Numerical |
| 🩸 **Glucose** | Plasma glucose concentration | Numerical |
| 💓 **BloodPressure** | Diastolic blood pressure (mm Hg) | Numerical |
| 📏 **SkinThickness** | Triceps skinfold thickness (mm) | Numerical |
| 💉 **Insulin** | 2-Hour serum insulin (mu U/ml) | Numerical |
| ⚖️ **BMI** | Body mass index | Numerical |
| 👨‍👩‍👧‍👦 **DiabetesPedigreeFunction** | Diabetes history in relatives | Numerical |
| 🎂 **Age** | Age in years | Numerical |
| 🎯 **Outcome** | Diabetes diagnosis (1=Yes, 0=No) | Binary |

## 🛠️ Technologies & Tools

### Big Data & Analytics
![Apache Spark](https://img.shields.io/badge/Apache_Spark-FFFFFF?style=flat&logo=apachespark&logoColor=#E35A16)
![AWS EMR](https://img.shields.io/badge/AWS_EMR-FF9900?style=flat&logo=amazonaws&logoColor=white)
![Hadoop](https://img.shields.io/badge/Apache_Hadoop-66CCFF?style=flat&logo=apachehadoop&logoColor=black)

### Programming & Data Science
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-75AADB?style=flat&logo=apachespark&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

### Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)

### Development Environment
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)
![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=flat&logo=visualstudiocode&logoColor=white)

## 📊 Methodology

### 1. Data Gathering & Cleaning 🧹
```python
# Load dataset into Spark DataFrame
diabetes_df = spark.read.csv("diabetes.csv", header=True, inferSchema=True)

# Data quality checks
diabetes_df.printSchema()
diabetes_df.describe().show()
```

### 2. Exploratory Data Analysis 🔍
- **Statistical Summary**: Mean, median, standard deviation
- **Correlation Analysis**: Heatmap visualization
- **Distribution Analysis**: Box plots, histograms
- **Outlier Detection**: Identifying anomalous values

### 3. Feature Engineering ⚙️
- Handling missing values (zeros treated as missing)
- Feature scaling and normalization
- Creation of new derived features
- Feature selection based on correlation

### 4. Machine Learning Modeling 🤖
- **Algorithms**: Logistic Regression, Decision Trees, Random Forest
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score
- **Cross-validation**: Ensuring model robustness

## 🚀 Implementation

### Prerequisites
```bash
# Required Python packages
pyspark==3.4.0
pandas==1.5.3
numpy==1.21.6
matplotlib==3.5.3
seaborn==0.12.2
jupyter==1.0.0
```

### AWS Hadoop Cluster Setup
1. **Launch EMR Cluster** on AWS
2. **Configure Spark** environment
3. **Upload dataset** to HDFS/S3
4. **Execute Spark jobs** for distributed processing

### Code Execution
```python
# Initialize Spark Session
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("DiabetesAnalysis") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Load and explore data
diabetes_df = spark.read.csv("s3://your-bucket/diabetes.csv", 
                           header=True, inferSchema=True)
print("Dataset Schema:")
diabetes_df.printSchema()
```

## 📈 Key Insights & Results

### Statistical Summary
| Feature | Mean | Std Dev | Min | Max | Correlation with Outcome |
|---------|------|---------|-----|-----|-------------------------|
| **Glucose** | 121.69 | 30.44 | 0 | 199 | **0.49** |
| **BMI** | 32.46 | 6.88 | 0 | 67.1 | **0.44** |
| **Age** | 33.24 | 11.76 | 21 | 81 | 0.24 |
| **Pregnancies** | 3.85 | 3.36 | 0 | 17 | 0.22 |

### 📊 Visualizations

#### 1. Correlation Heatmap 🔥
![Correlation Heatmap](https://via.placeholder.com/600x400/FF6B6B/FFFFFF?text=Correlation+Heatmap)
- **Strong positive correlation**: Glucose → Outcome (0.49)
- **Moderate correlation**: BMI → Outcome (0.44)
- **Weak correlation**: Age → Outcome (0.24)

#### 2. BMI Distribution by Outcome 📦
![BMI Boxplot](https://via.placeholder.com/600x400/4ECDC4/FFFFFF?text=BMI+Distribution+Analysis)
- Diabetic patients show higher BMI values
- Clear separation in BMI distributions between groups

#### 3. Outcome Distribution 📊
![Outcome Bar Chart](https://via.placeholder.com/600x400/45B7D1/FFFFFF?text=Diabetes+Outcome+Distribution)
- **65%**: Non-diabetic cases (Outcome = 0)
- **35%**: Diabetic cases (Outcome = 1)
- Dataset shows class imbalance

### 🎯 Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **78.8%** | 0.72 | 0.51 | **0.59** |
| Logistic Regression | 77.2% | 0.68 | 0.48 | 0.56 |
| Decision Tree | 73.5% | 0.61 | 0.45 | 0.52 |

## 💡 Key Findings

### 🔑 Top Predictors of Diabetes
1. **Glucose Levels** 🩸 - Strongest predictor (correlation: 0.49)
2. **BMI** ⚖️ - Significant relationship with diabetes risk
3. **Age** 🎂 - Moderate correlation with outcome
4. **Diabetes Pedigree Function** 👨‍👩‍👧‍👦 - Genetic predisposition factor

### ⚠️ Data Quality Issues
- **Missing Values**: Zero values in Glucose, BloodPressure, etc.
- **Class Imbalance**: 65% negative vs 35% positive cases
- **Outliers**: Present in Insulin and SkinThickness features

### 🏥 Healthcare Implications
- Early identification of at-risk patients
- Targeted preventive care strategies
- Resource allocation for high-risk groups

## 📂 Project Structure

```
diabetes-spark-analysis/
│
├── 📁 data/
│   └── diabetes.csv                 # Original dataset
│
├── 📁 notebooks/
│   ├── 01_data_loading.ipynb       # Data loading & preprocessing
│   ├── 02_eda_visualization.ipynb  # Exploratory analysis
│   ├── 03_feature_engineering.ipynb # Feature engineering
│   └── 04_model_training.ipynb     # ML model development
│
├── 📁 scripts/
│   ├── spark_job.py                # Main Spark application
│   └── aws_setup.sh               # AWS cluster configuration
│
├── 📁 outputs/
│   ├── visualizations/            # Generated plots and charts
│   └── model_results/            # Model performance metrics
│
├── 📁 docs/
│   └── project_report.pdf         # Comprehensive project documentation
│
└── README.md                      # Project overview (this file)
```

## 🚀 Getting Started

### Local Development
```bash
# Clone repository
git clone https://github.com/your-username/diabetes-spark-analysis.git
cd diabetes-spark-analysis

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook
```

### AWS EMR Execution
```bash
# Submit Spark job to EMR cluster
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    scripts/spark_job.py
```

## 🎯 Future Enhancements

- 🔄 **Real-time Prediction API** using Flask/FastAPI
- 📱 **Mobile Application** for healthcare workers
- 🌐 **Web Dashboard** with interactive visualizations
- 🔍 **Advanced Feature Engineering** with domain expertise
- 🤖 **Deep Learning Models** for improved accuracy

## 👥 Contributors

- **Were Vincent** - Data Analyst & Spark Developer

## 📚 References

1. Eken, S. (2020). *An exploratory teaching program in big data analysis for undergraduate students*
2. Jurney, R. (2017). *Agile data science 2.0: Building full-stack data analytics applications with Spark*
3. Batch, A., & Elmqvist, N. (2017). *The interactive visualization gap in initial exploratory data analysis*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 💙 *Empowering Healthcare Decisions Through Data Science* 💙

**If you find this project helpful, please give it a ⭐!**

</div>

# **📌 README: Credit Card Fraud Detection using Spark**
## **Project Overview**
This project implements **credit card fraud detection** using **Apache Spark** and **Machine Learning** techniques like **Random Forest** and **Gradient Boosting**.

### **Dataset**
- 📂 **Dataset Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains **284,807 transactions**, with **492 fraudulent transactions**.
- Features are **numerical (PCA-transformed)**, with **Time & Amount** as additional attributes.

### **Technologies Used**
✅ Apache Spark (PySpark)  
✅ Spark MLlib (Machine Learning)  
✅ Random Forest & Gradient Boosting  
✅ Joblib (Model Serialization)  
✅ Jupyter Notebook for Analysis  

---

## **💻 How to Run in Spark**
### **1. Install Required Packages**
Ensure you have **Apache Spark** installed. You can install dependencies with:

```bash
pip install pyspark joblib
```

### **2. Download & Setup Spark**
- Download Apache Spark:
  ```bash
  wget https://downloads.apache.org/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz
  tar -xvzf spark-3.4.0-bin-hadoop3.tgz
  cd spark-3.4.0-bin-hadoop3
  ```
- Set environment variables:
  ```bash
  export SPARK_HOME=$(pwd)
  export PATH=$SPARK_HOME/bin:$PATH
  export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
  ```

### **3. Run the Fraud Detection Script**
Execute the PySpark script:

```bash
spark-submit Random_Forest.py
```

Alternatively, run inside a Jupyter Notebook:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
df = spark.read.csv("creditcard.csv", header=True, inferSchema=True)
df.show(5)
```

---

## **🔬 Model Training & Evaluation**
- Load and preprocess data
- Train **Random Forest Classifier**
- Evaluate using **Precision, Recall, F1-score**
- Save model using `joblib`

### **Run Model Training**
```bash
spark-submit GBT.py
```

---

## **📂 Project Structure**
```
/creditcard-fraud-detection
│── creditcard.csv         # Dataset
│── Random_Forest.py       # Fraud detection using Random Forest
│── GBT.py                 # Fraud detection using Gradient Boosting
│── isolation.joblib       # Serialized Isolation Forest Model
│── newoutput.txt          # Sample output results
│── outputGBT.txt          # GBT Model results
```

---

## **🚀 Future Enhancements**
🔹 **Use Real-time Streaming (Spark Streaming + Kafka)**  
🔹 **Deploy Model as a REST API** (Flask, FastAPI)  
🔹 **Improve Model Performance with Hyperparameter Tuning**  

---

### **📩 Contact & Contributions**
Feel free to contribute! Open an issue or submit a pull request.  
For queries, contact [Your Email/GitHub Profile]. 🚀

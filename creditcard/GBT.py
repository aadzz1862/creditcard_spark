import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics

# Start the timer
start_time = time.time()

# Initialize a Spark session
spark = SparkSession.builder.appName("GBTForFraudDetection").getOrCreate()

# Load and prepare the dataset
data = spark.read.csv("creditcard.csv", header=True, inferSchema=True)

# Cast 'Class' column to DoubleType
data = data.withColumn("Class", col("Class").cast(DoubleType()))

# Feature columns and the target label
feature_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                   'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Assemble the features into a single feature vector column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# Rename the 'Class' column to 'label'
data = data.withColumnRenamed("Class", "label")

# Create a GBT model
gbt = GBTClassifier(labelCol="label", featuresCol="features")

# Create a parameter grid for tuning the model
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [5, 10])
             .addGrid(gbt.maxIter, [10, 20])
             .build())

# Create the evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")

# Set up the cross-validation
crossval = CrossValidator(estimator=gbt,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)  # Use 5 folds in cross-validation

# Fit the model
cvModel = crossval.fit(data)

# Get the best model
bestModel = cvModel.bestModel

# Make predictions with the best model
predictions = bestModel.transform(data)

# Evaluate the best model
accuracy = evaluator.evaluate(predictions)
f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(predictions)
precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)

# Confusion Matrix
predictionAndLabels = predictions.select("prediction", "label").rdd
metrics = MulticlassMetrics(predictionAndLabels)
confusionMatrix = metrics.confusionMatrix().toArray()

# Stop the timer
end_time = time.time()

# Calculate total execution time
total_time = end_time - start_time

# Write results to a file
output_file = 'outputGBT.txt'
try:
    with open(output_file, 'w') as file:
        file.write("Best Model's Parameters: {}\n".format(bestModel.extractParamMap()))
        file.write("Best Model's Accuracy: {}\n".format(accuracy))
        file.write("Best Model's F1 Score: {}\n".format(f1))
        file.write("Best Model's Precision: {}\n".format(precision))
        file.write("Best Model's Recall: {}\n".format(recall))
        file.write("Confusion Matrix:\n{}\n".format(confusionMatrix))
        file.write("Total Execution Time: {} seconds\n".format(total_time))
    print("Results written to {}".format(output_file))
except Exception as e:
    print("Failed to write to file: {}".format(output_file))
    traceback.print_exc()

# Stop the Spark session
spark.stop()

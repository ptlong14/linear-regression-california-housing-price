from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np

# ================== Spark Session ==================
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Linear Regression Full Analysis") \
    .getOrCreate()

# ================== Load dữ liệu ==================
df = spark.read.csv(
    "housing_clean.csv",
    header=True,
    inferSchema=True
)

# ================== Feature ==================
feature_columns = [
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms',
    'population', 'households',
    'median_income',
    'ocean_proximity_encoded'
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)
df = assembler.transform(df)

# ================== Chuẩn hóa ==================
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=True,
    withStd=True
)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# ================== Evaluator ==================
mse_eval  = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="mse")
rmse_eval = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="rmse")
r2_eval   = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="r2")
mae_eval  = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="mae")

mse_list, rmse_list, r2_list, mae_list = [], [], [], []

# ================== CHẠY 20 LẦN ĐÁNH GIÁ ==================
for i in range(1, 21):
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=i)

    lr = LinearRegression(
        featuresCol="scaledFeatures",
        labelCol="median_house_value",
        regParam=0.1,
        elasticNetParam=0.0
    )

    model = lr.fit(train_data)
    predictions = model.transform(test_data)

    mse_list.append(mse_eval.evaluate(predictions))
    rmse_list.append(rmse_eval.evaluate(predictions))
    r2_list.append(r2_eval.evaluate(predictions))
    mae_list.append(mae_eval.evaluate(predictions))

    print(
        f"Run {i:02d} | "
        f"MSE: {mse_list[-1]:.4f} | "
        f"RMSE: {rmse_list[-1]:.4f} | "
        f"R²: {r2_list[-1]:.4f} | "
        f"MAE: {mae_list[-1]:.4f}"
    )

print("\n=== TRUNG BÌNH ± ĐỘ LỆCH CHUẨN (20 RUNS) ===")
print(f"MSE:  {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
print(f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
print(f"R²:   {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
print(f"MAE:  {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")

# ================== TRAIN CUỐI + LƯU MÔ HÌNH ==================
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
final_model = lr.fit(train_data)

coef = final_model.coefficients.toArray()
intercept = final_model.intercept
mean = scaler_model.mean.toArray()
std = scaler_model.std.toArray()

np.save("coef.npy", coef)
np.save("intercept.npy", np.array([intercept]))
np.save("mean.npy", mean)
np.save("std.npy", std)
np.save("feature_columns.npy", np.array(feature_columns))

print("Train xong, mô hình và scaler đã được lưu.")

spark.stop()

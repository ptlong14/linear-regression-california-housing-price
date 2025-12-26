
from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import StringIndexer
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Phan Tich Mo Ta") \
    .getOrCreate()
df = spark.read.csv(
    "housing_clean.csv",
    header=True,
    inferSchema=True
)
num_cols = [c for c, t in df.dtypes if t in ("int", "double", "float", "bigint", "long")]

stats = []

for c in num_cols:
    # Count, Min, Max, Mean, Variance, Std
    row = df.select(
        F.count(c).alias("count"),
        F.min(c).alias("min"),
        F.max(c).alias("max"),
        F.mean(c).alias("mean"),
        F.variance(c).alias("variance"),
        F.stddev(c).alias("std")
    ).first()

    # Median, Q1, Q3
    q1, median, q3 = df.approxQuantile(c, [0.25, 0.5, 0.75], 0.01)
    iqr = q3 - q1

    # Mode (lấy giá trị xuất hiện nhiều nhất, kiểm tra None)
    mode_row = df.groupBy(c).count().orderBy(F.desc("count")).first()
    mode = mode_row[0] if mode_row else None

    stats.append((
        c,
        float(row["count"]) if row["count"] is not None else None,
        float(row["min"]) if row["min"] is not None else None,
        float(row["max"]) if row["max"] is not None else None,
        float(row["mean"]) if row["mean"] is not None else None,
        float(median) if median is not None else None,
        float(mode) if mode is not None else None,
        float(row["variance"]) if row["variance"] is not None else None,
        float(row["std"]) if row["std"] is not None else None,
        float(iqr) if iqr is not None else None
    ))

# Định nghĩa schema rõ ràng
schema = StructType([
    StructField("feature", StringType(), True),
    StructField("count", DoubleType(), True),
    StructField("min", DoubleType(), True),
    StructField("max", DoubleType(), True),
    StructField("mean", DoubleType(), True),
    StructField("median", DoubleType(), True),
    StructField("mode", DoubleType(), True),
    StructField("variance", DoubleType(), True),
    StructField("std_dev", DoubleType(), True),
    StructField("IQR", DoubleType(), True)
])

# Tạo DataFrame PySpark
stats_df = spark.createDataFrame(stats, schema=schema)

# Hiển thị kết quả
stats_df.show(truncate=False)

# Xuất CSV
stats_df.toPandas().to_csv("statistical_summary.csv", index=False)

#Biểu đồ cột ocean_proximity
cat_counts = (
    df.groupBy("ocean_proximity_encoded")
      .count()
      .orderBy(F.desc("count"))
)
pdf = cat_counts.toPandas()
plt.figure(figsize=(8,5))
plt.bar(pdf["ocean_proximity_encoded"], pdf["count"])
plt.xlabel("Ocean Proximity")
plt.ylabel("Number of Records")
plt.title("Distribution of Ocean Proximity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Histogram all
df_plot = df.select(num_cols)
pdf = df_plot.toPandas()
pdf[num_cols].hist(figsize=(14,10), bins=40)
plt.suptitle("Numerical Features Distribution")
plt.show()

#Histogram và boxplot của median house value
pandas_df_price_income = df.select("median_house_value").toPandas()
plt.figure(figsize=(8,5))
sns.histplot(pandas_df_price_income["median_house_value"], bins=50, kde=True)
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.title("Distribution of House Prices")
plt.show()
plt.figure(figsize=(6,3))
sns.boxplot(x=pandas_df_price_income["median_house_value"])
plt.title("Boxplot of Median House Value")
plt.show()

#Histogram và boxplot của longitude
pandas_df_longitude = df.select("longitude").toPandas()
plt.figure(figsize=(8,5))
sns.histplot(pandas_df_longitude["longitude"], bins=50, kde=True)
plt.xlabel("Longitude")
plt.ylabel("Frequency")
plt.title("Distribution of longitude")
plt.show()
plt.figure(figsize=(6,3))
sns.boxplot(x=pandas_df_longitude["longitude"])
plt.title("Boxplot of Longitude")
plt.show()

#Histogram và boxplot của latitude
pandas_df_latitude = df.select("latitude").toPandas()
plt.figure(figsize=(8,5))
sns.histplot(pandas_df_latitude["latitude"], bins=50, kde=True)
plt.xlabel("Latitude")
plt.ylabel("Frequency")
plt.title("Distribution of latitude")
plt.show()
plt.figure(figsize=(6,3))
sns.boxplot(x=pandas_df_latitude["latitude"])
plt.title("Boxplot of Latitude")
plt.show()

# Scatter House Prices - Income
pandas_df_price_income = df.select("median_income", "median_house_value").toPandas()
plt.figure(figsize=(8,5))
sns.scatterplot(x=pandas_df_price_income["median_income"], y=pandas_df_price_income["median_house_value"], alpha=0.5)
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("House Prices by Income")
plt.show()

#Scatter Geographical Location - House Prices
pandas_df_geo_price = df.select("longitude", "latitude", "median_house_value").toPandas()
plt.figure(figsize=(10,6))
plt.scatter(pandas_df_geo_price["longitude"], pandas_df_geo_price["latitude"], c=pandas_df_geo_price["median_house_value"], cmap="coolwarm", alpha=0.5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("House Prices by Geographic Location")
plt.colorbar(label="House Price")
plt.show()

#Chọn các cột numerical + cột encoded để tính correlation
feature_cols = [col for col, dtype in df.dtypes if dtype in ['int', 'double', 'float']]

#Assemble thành vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df)

#Tính correlation matrix (Pearson)
corr_matrix = Correlation.corr(df_assembled, "features", "pearson").head()[0].toArray()

# Chuyển sang pandas DataFrame để visualize bằng seaborn
corr_pd = pd.DataFrame(corr_matrix, columns=feature_cols, index=feature_cols)

# Vẽ heatmap
plt.figure(figsize=(15,12))
sns.heatmap(corr_pd, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

spark.stop()
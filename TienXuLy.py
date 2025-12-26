from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession

from pyspark.sql import functions as F

#Khởi tạo Spark Session
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Tien Xu Ly") \
    .getOrCreate()
#Đọc csv
df = spark.read.csv(
    "housing_raw.csv",
    header=True,
    inferSchema=True
)
#Xem 5 dòng đầu
df.show(5)
#Thông tin schema
df.printSchema()

#Thống kê mô tả
df.describe().show()

# Đếm số giá trị null mỗi cột
from pyspark.sql.functions import col, isnan, when, count
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# Điền khuyết cột missing data bằng mean
df = df.fillna({'total_bedrooms': df.selectExpr("mean(total_bedrooms)").collect()[0][0]})
#Check lại
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

indexer = StringIndexer(inputCol="ocean_proximity", outputCol="ocean_proximity_encoded")
df = indexer.fit(df).transform(df)
df = df.drop("ocean_proximity")
df.show(5)

#Xử lý giá trị ngoại lai
def cap_outlier_iqr(df, column_name, threshold=1.5):
    Q1, Q3 = df.approxQuantile(
        column_name,
        [0.25, 0.75],
        0.01
    )

    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    return df.withColumn(
        column_name,
        when(col(column_name) < lower_bound, lower_bound)
        .when(col(column_name) > upper_bound, upper_bound)
        .otherwise(col(column_name))
    )


# Áp dụng cho cột target (giá nhà)
feature_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
for col_name in feature_columns:
    df = cap_outlier_iqr(df, col_name)

#Kiểm tra kích thước DataFrame sau lọc outlier
print("Số dòng:", df.count())
print("Số cột:", len(df.columns))
print("\n")

pdf_clean = df.toPandas()
pdf_clean.to_csv(
    "housing_clean.csv",
    index=False,
    header=True
)
spark.stop()
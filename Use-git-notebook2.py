# Databricks notebook source
# MAGIC %scala
# MAGIC print(res7)

# COMMAND ----------

spark.conf.get("spark.app.id")

# COMMAND ----------

print("helloworld")
print(spark)
spark.range(1000).repartition(10).collect()

# COMMAND ----------



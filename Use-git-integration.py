# Databricks notebook source
print("helloworld")
print(spark)
spark.range(1000).repartition(10).collect()

# COMMAND ----------

# MAGIC %scala 
# MAGIC val a = 10

# COMMAND ----------

# MAGIC %scala
# MAGIC print(spark)
# MAGIC spark.createDataset((0 to 100000).toList).repartition(100).collect()

# COMMAND ----------

# MAGIC %scala
# MAGIC print("hello")

# COMMAND ----------

# MAGIC %scala
# MAGIC spark.conf.get("spark.app.id")

# COMMAND ----------

spark.conf.get("spark.app.id")

# COMMAND ----------

# MAGIC %scala
# MAGIC print(res7)

# COMMAND ----------

# MAGIC %scala
# MAGIC print(res7)

# COMMAND ----------

help(spark)

# COMMAND ----------



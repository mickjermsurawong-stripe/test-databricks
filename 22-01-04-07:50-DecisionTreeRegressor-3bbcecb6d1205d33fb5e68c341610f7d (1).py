# Databricks notebook source
# MAGIC %md
# MAGIC # Decision Tree Regressor training
# MAGIC This is an auto-generated notebook. To reproduce these results, attach this notebook to the **mickjermsurawong-adhoc-cluster** cluster and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/1626878219706274/s?orderByKey=metrics.%60val_r2_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/1626878219706272) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.
# MAGIC 
# MAGIC Runtime Version: _9.1.x-cpu-ml-scala2.12_

# COMMAND ----------

import mlflow
import databricks.automl_runtime

# Use MLflow to track experiments
mlflow.set_experiment("/Users/mickjermsurawong@stripe.com/databricks_automl/dropoff_is_weekend_trip_dropoff_features-2022_01_04-14_49")

target_col = "dropoff_is_weekend"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

from mlflow.tracking import MlflowClient
import os
import uuid
import shutil
import pandas as pd

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)


# Download the artifact and read it into a pandas DataFrame
input_client = MlflowClient()
input_data_path = input_client.download_artifacts("ce8e936ddebe46e9b44c83943eff358e", "data", input_temp_dir)

df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

transformers = []

# COMMAND ----------

# MAGIC %md
# MAGIC %md ### Datetime Preprocessor
# MAGIC For each datetime column, extract relevant information from the date:
# MAGIC - Unix timestamp
# MAGIC - whether the date is a weekend
# MAGIC - whether the date is a holiday
# MAGIC 
# MAGIC Additionally, extract extra information from columns with timestamps:
# MAGIC - hour of the day (one-hot encoded)
# MAGIC 
# MAGIC For cyclic features, plot the values along a unit circle to encode temporal proximity:
# MAGIC - hour of the day
# MAGIC - hours since the beginning of the week
# MAGIC - hours since the beginning of the month
# MAGIC - hours since the beginning of the year

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from databricks.automl_runtime.sklearn import TimestampTransformer

for col in ["yyyy_mm"]:
    timestamp_transformer = TimestampTransformer()
    ohe_transformer = ColumnTransformer(
        [("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"), [timestamp_transformer.HOUR_COLUMN_INDEX])],
        remainder="passthrough")
    timestamp_preprocessor = Pipeline([
        ("extractor", timestamp_transformer),
        ("onehot_encoder", ohe_transformer)
    ])
    transformers.append((f"timestamp_{col}", timestamp_preprocessor, [col]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC 
# MAGIC Missing values for numerical columns are imputed with mean for consistency

# COMMAND ----------

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputer", SimpleImputer(strategy="mean"))
])

transformers.append(("numerical", numerical_pipeline, ["count_trips_window_30m_dropoff_zip", "ts", "zip"]))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature standardization
# MAGIC Scale all feature columns to be centered around zero with unit variance.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training - Validation Split
# MAGIC Split the input data into training and validation data

# COMMAND ----------

from sklearn.model_selection import train_test_split

split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

X_train, X_val, y_train, y_val = train_test_split(split_X, split_y, random_state=29617910)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train regression model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/1626878219706274/s?orderByKey=metrics.%60val_r2_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor

help(DecisionTreeRegressor)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(display='diagram')

skdt_regressor = DecisionTreeRegressor(
  criterion="mse",
  max_features=1.0,
  min_samples_leaf=13,
  min_samples_split=13,
  random_state=29617910,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("regressor", skdt_regressor),
])

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="decision_tree_regressor") as mlflow_run:
    model.fit(X_train, y_train)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    skdt_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val,
                                                                prefix="val_")
    display(pd.DataFrame(skdt_val_metrics, index=[0]))

# COMMAND ----------

# Patch requisite packages to the model environment YAML for model serving
import os
import shutil
import uuid
import yaml

None

import holidays
from mlflow.tracking import MlflowClient

skdt_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(skdt_temp_dir)
skdt_client = MlflowClient()
skdt_model_env_path = skdt_client.download_artifacts(mlflow_run.info.run_id, "model/conda.yaml", skdt_temp_dir)
skdt_model_env_str = open(skdt_model_env_path)
skdt_parsed_model_env_str = yaml.load(skdt_model_env_str, Loader=yaml.FullLoader)

skdt_parsed_model_env_str["dependencies"][-1]["pip"].append(f"holidays=={holidays.__version__}")

with open(skdt_model_env_path, "w") as f:
  f.write(yaml.dump(skdt_parsed_model_env_str))
skdt_client.log_artifact(run_id=mlflow_run.info.run_id, local_path=skdt_model_env_path, artifact_path="model")
shutil.rmtree(skdt_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC 
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

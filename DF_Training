#!/usr/bin/env python3

import os
import mlflow
import mlflow.spark

from mlflow import log_metric, log_param, log_artifact
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

#Current Version V1 

def main():
    # Used Data Set
    # Download this from https://raw.githubusercontent.com/cartershanklin/pyspark-cheatsheet/master/data/auto-mpg-fixed.csv
    INPUT_PATH = "oci://bucket@namespace/ML_Training/training-data/file-name.csv"

    # Set up MLFlow
    remote_server_uri = "http://mlflow-server-url.oraclevcn.com:5000"  # set to your server URI
    mlflow.set_tracking_uri(remote_server_uri)
    experiment_name = 'car-mileage-model-electric'
    experiment_id = mlflow.get_experiment_by_name(experiment_name)
    if experiment_id is None:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

    print('Experiment:', experiment_id)

    # Use autologging
    mlflow.pyspark.ml.autolog()

    # Set up Spark.
    spark_session = get_dataflow_spark_session()

    # Load our data.
    print("Reading data from object store")
    df = spark_session.read.option("header", "true").csv(INPUT_PATH)

    # Convert some columns into numbers.
    print("Converting strings to numbers")
    for (column_name) in "mpg cylinders displacement horsepower weight acceleration electric".split():
        df = df.withColumn(column_name, col(column_name).cast("double")
    )
    df = df.withColumn("modelyear", col("modelyear").cast("int"))
    df = df.withColumn("origin", col("origin").cast("int"))

    #Prepare features



    #Convert text to vectors for feature extraction
    vectorAssembler = VectorAssembler(
        inputCols=[
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "electric",
        ],
        outputCol="features",
        handleInvalid="skip",
    )
    assembled = vectorAssembler.transform(df)
    assembled = assembled.select(["features", "mpg", "carname"])

    # Random test/train split 30/70.
    train_df, test_df = assembled.randomSplit([0.7, 0.3])

    # Start the run
    with mlflow.start_run():
        # Define the model.
        rf = RandomForestRegressor(numTrees=50, featuresCol="features", labelCol="mpg",)

        # Train the model.
        print("Training the model.")
        rf_model = rf.fit(train_df)

        # Log parameters
        #mlflow.log_param("labelCol", rf.getLabelCol())
        #mlflow.log_param("numTrees", rf.getNumTrees())

        # Make predictions.
        predictions = rf_model.transform(test_df)
        predictions.select(rf_model.getPredictionCol()).show(10)

        # Evaluate the model.
        r2 = RegressionEvaluator(
            labelCol="mpg", predictionCol="prediction", metricName="r2"
        ).evaluate(predictions)
        rmse = RegressionEvaluator(
            labelCol="mpg", predictionCol="prediction", metricName="rmse"
        ).evaluate(predictions)
        print("RMSE={} r2={}".format(rmse, r2))

        # Log metrics
        #mlflow.log_metric("rmse", rmse)
        #mlflow.log_metric("r2", r2)

        # Log Model
        #mlflow.spark.log_model(rf_model, "rf-spark-model")

        print(f"Logged model with RMSE: {rmse} and R2: {r2}")

    #Save the model
    model_path = "oci://bucket@namespace/ML_Training/models/auto-mpg-fixed-electric"
    rf_model.write().overwrite().save(model_path)

    # Convert metrics to a DataFrame
    eval_df = spark_session.createDataFrame([
        ("r2", r2),
        ("rmse", rmse)
    ], ["metric", "value"])

    # Save to CSV
    output_path_eval = "oci://bucket@namespace/ML_Training/output/eval-v4"
    eval_df.write.mode("overwrite").option("header", True).csv(output_path_eval)


def get_dataflow_spark_session(
    app_name="DataFlow", file_location=None, profile_name=None, spark_config={}
):
    """
    Get a Spark session in a way that supports running locally or in Data Flow.
    """
    if in_dataflow():
        spark_builder = SparkSession.builder.appName(app_name)
    else:
        # Import OCI.
        try:
            import oci
        except:
            raise Exception(
                "You need to install the OCI python library to test locally"
            )

        # Use defaults for anything unset.
        if file_location is None:
            file_location = oci.config.DEFAULT_LOCATION
        if profile_name is None:
            profile_name = oci.config.DEFAULT_PROFILE

        # Load the config file.
        try:
            oci_config = oci.config.from_file(
                file_location=file_location, profile_name=profile_name
            )
        except Exception as e:
            print("You need to set up your OCI config properly to run locally")
            raise e
        conf = SparkConf()
        conf.set("fs.oci.client.auth.tenantId", oci_config["tenancy"])
        conf.set("fs.oci.client.auth.userId", oci_config["user"])
        conf.set("fs.oci.client.auth.fingerprint", oci_config["fingerprint"])
        conf.set("fs.oci.client.auth.pemfilepath", oci_config["key_file"])
        conf.set(
            "fs.oci.client.hostname",
            "https://objectstorage.{0}.oraclecloud.com".format(oci_config["region"]),
        )
        spark_builder = SparkSession.builder.appName(app_name).config(conf=conf)

    # Add in extra configuration.
    for key, val in spark_config.items():
        spark_builder.config(key, val)

    # Create the Spark session.
    session = spark_builder.getOrCreate()
    return session


def get_authenticated_client(token_path, client, file_location=None, profile_name=None):
    """
    Get an an authenticated OCI client.
    Example: get_authenticated_client(token_path, oci.object_storage.ObjectStorageClient)
    """
    import oci

    if not in_dataflow():
        # We are running locally, use our API Key.
        if file_location is None:
            file_location = oci.config.DEFAULT_LOCATION
        if profile_name is None:
            profile_name = oci.config.DEFAULT_PROFILE
        config = oci.config.from_file(
            file_location=file_location, profile_name=profile_name
        )
        authenticated_client = client(config)
    else:
        # We are running in Data Flow, use our Delegation Token.
        with open(token_path) as fd:
            delegation_token = fd.read()
        signer = oci.auth.signers.InstancePrincipalsDelegationTokenSigner(
            delegation_token=delegation_token
        )
        authenticated_client = client(config={}, signer=signer)
    return authenticated_client


def in_dataflow():
    """
    Determine if we are running in OCI Data Flow by checking the environment.
    """
    if os.environ.get("HOME") == "/home/dataflow":
        return True
    return False


def get_delegation_token_path(spark):
    """
    Get the delegation token path when we're running in Data Flow.
    """
    if not in_dataflow():
        return None
    token_key = "spark.hadoop.fs.oci.client.auth.delegationTokenPath"
    token_path = spark.sparkContext.getConf().get(token_key)
    if not token_path:
        raise Exception(f"{token_key} is not set")
    return token_path


def get_temporary_directory():
    if in_dataflow():
        return "/opt/spark/work-dir/"
    else:
        import tempfile

        return tempfile.gettempdir()


if __name__ == "__main__":
    main()

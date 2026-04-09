"""
Data preprocessing functions for the fraud detection pipeline.
Covers data loading, cleaning, feature engineering, categorical encoding,
outlier handling, and train/test splitting.
"""

import logging
from typing import List, Dict, Optional, Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder

logger = logging.getLogger(__name__)


def load_raw_data(spark: SparkSession, data_path: str) -> DataFrame:
    """
    Load raw CSV data into a PySpark DataFrame.

    Args:
        spark: Active SparkSession
        data_path: Path to the raw CSV file

    Returns:
        PySpark DataFrame with raw data loaded
    """
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
    from spark_utils import load_dataframe, get_dataframe_info

    logger.info(f"\n{'='*60}")
    logger.info("STEP 1 - LOADING RAW DATA")
    logger.info(f"{'='*60}")
    logger.info(f"  Source: {data_path}")

    df = load_dataframe(spark, data_path, format="csv")

    info = get_dataframe_info(df)
    logger.info(f"  Rows: {info['num_rows']} | Columns: {info['num_columns']}")
    logger.info(f"  Schema: {df.dtypes}")

    return df


def clean_data(df: DataFrame, target_column: str) -> DataFrame:
    """
    Perform basic data cleaning: drop duplicates, drop nulls in target,
    and cast column types according to config.

    Args:
        df: Raw PySpark DataFrame
        target_column: Name of the label/target column

    Returns:
        Cleaned PySpark DataFrame
    """
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
    from spark_utils import check_missing_values
    from config import get_columns

    logger.info(f"\n{'='*60}")
    logger.info("STEP 2 - DATA CLEANING")
    logger.info(f"{'='*60}")

    initial_rows = df.count()

    # Drop duplicate rows
    df = df.dropDuplicates()
    logger.info(f"  Dropped duplicates: {initial_rows - df.count()} rows removed")

    # Drop rows where target is null
    before_target_drop = df.count()
    df = df.filter(F.col(target_column).isNotNull())
    logger.info(f"  Dropped null target rows: {before_target_drop - df.count()} rows removed")

    # Cast target column to integer
    df = df.withColumn(target_column, F.col(target_column).cast("integer"))

    # Cast numeric columns to double
    columns_config = get_columns()
    numeric_columns = columns_config.get("numeric_columns", [])
    for col_name in numeric_columns:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast("double"))
            logger.info(f"  Cast '{col_name}' to double")

    # Report missing values
    missing = check_missing_values(df)
    total_missing = sum(missing.values())
    if total_missing > 0:
        logger.info(f"  Missing values detected: {missing}")
    else:
        logger.info("  No missing values found")

    logger.info(f"  Cleaned dataset: {df.count()} rows remaining")
    return df


def engineer_features(df: DataFrame) -> DataFrame:
    """
    Apply feature engineering steps:
      - Parse and extract time-based features from signup_time and purchase_time
      - Compute seconds elapsed between signup and purchase
      - Drop raw timestamp and high-cardinality ID columns

    Args:
        df: Cleaned PySpark DataFrame

    Returns:
        PySpark DataFrame with engineered features
    """
    logger.info(f"\n{'='*60}")
    logger.info("STEP 3 - FEATURE ENGINEERING")
    logger.info(f"{'='*60}")

    # Parse timestamps
    if "signup_time" in df.columns:
        df = df.withColumn("signup_time", F.to_timestamp("signup_time"))
    if "purchase_time" in df.columns:
        df = df.withColumn("purchase_time", F.to_timestamp("purchase_time"))

    # Time delta: seconds between signup and purchase
    if "signup_time" in df.columns and "purchase_time" in df.columns:
        df = df.withColumn(
            "time_to_purchase_secs",
            F.unix_timestamp("purchase_time") - F.unix_timestamp("signup_time")
        )
        logger.info("  Created feature: 'time_to_purchase_secs'")

    # Extract hour and day-of-week from purchase timestamp
    if "purchase_time" in df.columns:
        df = df.withColumn("purchase_hour", F.hour("purchase_time"))
        df = df.withColumn("purchase_day_of_week", F.dayofweek("purchase_time"))
        logger.info("  Created features: 'purchase_hour', 'purchase_day_of_week'")

    # Drop raw timestamp columns
    timestamp_cols = [c for c in ["signup_time", "purchase_time"] if c in df.columns]
    if timestamp_cols:
        df = df.drop(*timestamp_cols)
        logger.info(f"  Dropped raw timestamp columns: {timestamp_cols}")

    # Drop high-cardinality ID columns
    id_cols = [c for c in ["user_id", "device_id", "ip_address"] if c in df.columns]
    if id_cols:
        df = df.drop(*id_cols)
        logger.info(f"  Dropped ID/high-cardinality columns: {id_cols}")

    logger.info(f"  Feature engineering complete. Columns: {df.columns}")
    return df


def encode_categorical_features(
    df: DataFrame,
    nominal_columns: List[str],
    target_column: str
) -> Tuple[DataFrame, Optional[PipelineModel]]:
    """
    Encode categorical (nominal) columns using StringIndexer + OneHotEncoder
    via a fitted PySpark ML Pipeline.

    Args:
        df: PySpark DataFrame
        nominal_columns: List of categorical column names to encode
        target_column: Target column name (excluded from encoding)

    Returns:
        Tuple of (encoded DataFrame, fitted PipelineModel or None)
    """
    logger.info(f"\n{'='*60}")
    logger.info("STEP 4 - CATEGORICAL ENCODING")
    logger.info(f"{'='*60}")

    cols_to_encode = [c for c in nominal_columns if c in df.columns and c != target_column]
    logger.info(f"  Encoding columns: {cols_to_encode}")

    if not cols_to_encode:
        logger.info("  No categorical columns to encode - skipping")
        return df, None

    stages = []
    for col_name in cols_to_encode:
        index_col = f"{col_name}_index"
        ohe_col   = f"{col_name}_ohe"

        indexer = StringIndexer(
            inputCol=col_name,
            outputCol=index_col,
            handleInvalid="keep"
        )
        encoder = OneHotEncoder(
            inputCols=[index_col],
            outputCols=[ohe_col],
            handleInvalid="keep"
        )
        stages += [indexer, encoder]

    pipeline = Pipeline(stages=stages)
    model    = pipeline.fit(df)
    df_enc   = model.transform(df)

    # Drop original string columns and intermediate index columns
    cols_to_drop = []
    for col_name in cols_to_encode:
        cols_to_drop.extend([col_name, f"{col_name}_index"])
    df_enc = df_enc.drop(*cols_to_drop)

    logger.info(f"  Encoding complete. OHE columns: {[c for c in df_enc.columns if c.endswith('_ohe')]}")
    return df_enc, model


def split_data(
    df: DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Split the DataFrame into train and test sets, then separate features from labels.

    Args:
        df: Preprocessed PySpark DataFrame
        target_column: Name of the label column
        test_size: Fraction of data to hold out as the test set (0 < test_size < 1)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, Y_train, Y_test) PySpark DataFrames
    """
    logger.info(f"\n{'='*60}")
    logger.info("STEP 5 - TRAIN/TEST SPLIT")
    logger.info(f"{'='*60}")
    logger.info(f"  test_size={test_size} | random_state={random_state}")

    train_ratio = 1.0 - test_size
    train_df, test_df = df.randomSplit([train_ratio, test_size], seed=random_state)

    train_df.cache()
    test_df.cache()

    logger.info(f"  Train rows : {train_df.count()}")
    logger.info(f"  Test rows  : {test_df.count()}")

    feature_columns = [c for c in df.columns if c != target_column]

    X_train = train_df.select(feature_columns)
    X_test  = test_df.select(feature_columns)
    Y_train = train_df.select(target_column)
    Y_test  = test_df.select(target_column)

    logger.info("  Train class distribution:")
    train_df.groupBy(target_column).count().orderBy(target_column).show()

    return X_train, X_test, Y_train, Y_test

"""
Data preprocessing functions for the fraud detection pipeline.
Covers data loading, cleaning, missing value imputation, feature engineering,
outlier handling, feature binning, categorical encoding, scaling, and splitting.
"""

import os
import sys
import logging
from typing import List, Dict, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from spark_utils import load_dataframe, get_dataframe_info, check_missing_values

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from feature_binning import BucketizerBinningStrategy

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
    logger.info(f"\n{'='*60}")
    logger.info("STEP 1 - LOADING RAW DATA")
    logger.info(f"{'='*60}")
    logger.info(f"  Source: {data_path}")

    df = load_dataframe(spark, data_path, format="csv")

    info = get_dataframe_info(df)
    logger.info(f"  Rows: {info['num_rows']} | Columns: {info['num_columns']}")
    logger.info(f"  Schema: {df.dtypes}")

    return df


def clean_data(df: DataFrame, target_column: str, numeric_columns: List[str]) -> DataFrame:
    """
    Perform basic data cleaning: drop duplicates, drop nulls in target,
    cast numeric columns to double, and report missing values.

    Args:
        df: Raw PySpark DataFrame
        target_column: Name of the label/target column
        numeric_columns: List of numeric column names to cast to double

    Returns:
        Cleaned PySpark DataFrame
    """
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
    for col_name in numeric_columns:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast("double"))
            logger.info(f"  Cast '{col_name}' to double")

    logger.info(f"  Cleaned dataset: {df.count()} rows remaining")
    return df


def impute_missing_values(
    df: DataFrame,
    numeric_columns: List[str],
    nominal_columns: List[str]
) -> DataFrame:
    """
    Impute missing values in numeric and categorical columns.
    Numeric columns are filled with their median; categorical columns
    are filled with their mode (most frequent value).

    Args:
        df: PySpark DataFrame
        numeric_columns: Numeric column names to impute with median
        nominal_columns: Categorical column names to impute with mode

    Returns:
        PySpark DataFrame with missing values imputed
    """
    logger.info(f"\n{'='*60}")
    logger.info("STEP 3 - MISSING VALUE IMPUTATION")
    logger.info(f"{'='*60}")

    missing = check_missing_values(df)
    total_missing = sum(missing.values())

    if total_missing == 0:
        logger.info("  No missing values found - skipping imputation")
        return df

    logger.info(f"  Missing value counts: {missing}")

    # Impute numeric columns with median
    for col_name in numeric_columns:
        if col_name not in df.columns:
            continue
        if missing.get(col_name, 0) == 0:
            continue

        median_val = df.approxQuantile(col_name, [0.5], 0.01)[0]
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name).isNull() | F.isnan(col_name), median_val)
            .otherwise(F.col(col_name))
        )
        logger.info(f"  Imputed '{col_name}' with median={median_val:.4f}")

    # Impute categorical columns with mode
    for col_name in nominal_columns:
        if col_name not in df.columns:
            continue
        if missing.get(col_name, 0) == 0:
            continue

        mode_row = (
            df.filter(F.col(col_name).isNotNull() & (F.col(col_name) != ""))
            .groupBy(col_name)
            .count()
            .orderBy(F.desc("count"))
            .first()
        )
        if mode_row:
            mode_val = mode_row[col_name]
            df = df.withColumn(
                col_name,
                F.when(F.col(col_name).isNull() | (F.col(col_name) == ""), mode_val)
                .otherwise(F.col(col_name))
            )
            logger.info(f"  Imputed '{col_name}' with mode='{mode_val}'")

    logger.info("  Imputation complete")
    return df


def engineer_features(df: DataFrame, drop_columns: List[str]) -> DataFrame:
    """
    Apply feature engineering steps:
      - Parse and extract time-based features from signup_time and purchase_time
      - Compute seconds elapsed between signup and purchase
      - Drop raw timestamp columns and any configured ID/unused columns

    Args:
        df: Cleaned PySpark DataFrame
        drop_columns: List of column names to drop (e.g. user_id)

    Returns:
        PySpark DataFrame with engineered features
    """
    logger.info(f"\n{'='*60}")
    logger.info("STEP 4 - FEATURE ENGINEERING")
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

    # Drop configured ID/unused columns
    cols_to_drop = [c for c in drop_columns if c in df.columns]
    if cols_to_drop:
        df = df.drop(*cols_to_drop)
        logger.info(f"  Dropped configured columns: {cols_to_drop}")

    logger.info(f"  Feature engineering complete. Columns: {df.columns}")
    return df


def bin_features(df: DataFrame, binning_config: Dict) -> Tuple[DataFrame, List[str]]:
    """
    Bin numeric columns using BucketizerBinningStrategy.
    Each binned column replaces the original with a string label column
    named '{column}Bins' that will be encoded downstream.

    Args:
        df: PySpark DataFrame
        binning_config: Dictionary of column -> {splits, labels} from config

    Returns:
        Tuple of (DataFrame with binned columns, list of new bin column names)
    """
    logger.info(f"\n{'='*60}")
    logger.info("STEP 5 - FEATURE BINNING")
    logger.info(f"{'='*60}")

    bin_columns_created = []

    for col_name, col_cfg in binning_config.items():
        if col_name not in df.columns:
            logger.info(f"  Column '{col_name}' not found - skipping")
            continue

        splits = [float(s) for s in col_cfg.get("splits", [])]
        labels = col_cfg.get("labels", None)

        if len(splits) < 2:
            logger.info(f"  Invalid splits for '{col_name}' - skipping")
            continue

        strategy = BucketizerBinningStrategy(
            splits=splits,
            labels=labels,
            handle_invalid="keep"
        )
        df = strategy.bin_feature(df, col_name)
        bin_col = f"{col_name}Bins"
        bin_columns_created.append(bin_col)
        logger.info(f"  Binned '{col_name}' -> '{bin_col}'")

    logger.info(f"  Binning complete. New columns: {bin_columns_created}")
    return df, bin_columns_created


def encode_categorical_features(
    df: DataFrame,
    nominal_columns: List[str],
    bin_columns: List[str],
    target_column: str
) -> Tuple[DataFrame, Optional[PipelineModel]]:
    """
    Encode categorical (nominal) columns and any string bin columns using
    StringIndexer + OneHotEncoder via a fitted PySpark ML Pipeline.

    Args:
        df: PySpark DataFrame
        nominal_columns: List of categorical column names from config
        bin_columns: List of binned string column names (e.g. 'ageBins')
        target_column: Target column name (excluded from encoding)

    Returns:
        Tuple of (encoded DataFrame, fitted PipelineModel or None)
    """
    logger.info(f"\n{'='*60}")
    logger.info("STEP 6 - CATEGORICAL ENCODING")
    logger.info(f"{'='*60}")

    all_cat_columns = nominal_columns + bin_columns
    cols_to_encode = [
        c for c in all_cat_columns
        if c in df.columns and c != target_column
    ]
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

    ohe_cols = [c for c in df_enc.columns if c.endswith("_ohe")]
    logger.info(f"  Encoding complete. OHE columns: {ohe_cols}")
    return df_enc, model


def split_data(
    df: DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[DataFrame, DataFrame]:
    """
    Split the DataFrame into train and test sets.
    Returns full DataFrames (features + label) so that scaling can be
    fitted on the training set before separating X and Y.

    Args:
        df: Preprocessed PySpark DataFrame
        target_column: Name of the label column
        test_size: Fraction to hold out as test set (0 < test_size < 1)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df) PySpark DataFrames
    """
    logger.info(f"\n{'='*60}")
    logger.info("STEP 7 - TRAIN/TEST SPLIT")
    logger.info(f"{'='*60}")
    logger.info(f"  test_size={test_size} | random_state={random_state}")

    train_ratio = 1.0 - test_size
    train_df, test_df = df.randomSplit([train_ratio, test_size], seed=random_state)

    train_df.cache()
    test_df.cache()

    logger.info(f"  Train rows : {train_df.count()}")
    logger.info(f"  Test rows  : {test_df.count()}")

    logger.info("  Train class distribution:")
    train_df.groupBy(target_column).count().orderBy(target_column).show()

    return train_df, test_df


def scale_features(
    train_df: DataFrame,
    test_df: DataFrame,
    scale_columns: List[str]
) -> Tuple[DataFrame, DataFrame]:
    """
    Apply StandardScaler (z-score normalization) to numeric columns.
    Statistics are computed on the training set only and applied to both
    train and test to prevent data leakage.

    Args:
        train_df: Training PySpark DataFrame
        test_df: Test PySpark DataFrame
        scale_columns: List of numeric column names to scale

    Returns:
        Tuple of (scaled_train_df, scaled_test_df)
    """
    logger.info(f"\n{'='*60}")
    logger.info("STEP 8 - FEATURE SCALING")
    logger.info(f"{'='*60}")

    cols_to_scale = [c for c in scale_columns if c in train_df.columns]

    if not cols_to_scale:
        logger.info("  No valid scaling columns found - skipping")
        return train_df, test_df

    logger.info(f"  Scaling columns (standard scaler): {cols_to_scale}")

    # Compute mean and stddev from training set only
    stats_exprs = []
    for col_name in cols_to_scale:
        stats_exprs.append(F.mean(F.col(col_name)).alias(f"{col_name}_mean"))
        stats_exprs.append(F.stddev(F.col(col_name)).alias(f"{col_name}_std"))

    stats_row = train_df.select(stats_exprs).collect()[0]

    # Apply z-score: (x - mean) / std to both train and test
    for col_name in cols_to_scale:
        mean_val = stats_row[f"{col_name}_mean"]
        std_val  = stats_row[f"{col_name}_std"]

        if std_val is None or std_val == 0:
            logger.info(f"  Skipping '{col_name}' (std=0 or null)")
            continue

        scaled_expr = (F.col(col_name) - mean_val) / std_val

        train_df = train_df.withColumn(col_name, scaled_expr)
        test_df  = test_df.withColumn(col_name, scaled_expr)

        logger.info(f"  Scaled '{col_name}' (mean={mean_val:.4f}, std={std_val:.4f})")

    logger.info("  Scaling complete")
    return train_df, test_df


def separate_features_labels(
    train_df: DataFrame,
    test_df: DataFrame,
    target_column: str
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Separate feature columns from the label column in train and test sets.

    Args:
        train_df: Scaled training PySpark DataFrame
        test_df: Scaled test PySpark DataFrame
        target_column: Name of the label column

    Returns:
        Tuple of (X_train, X_test, Y_train, Y_test) PySpark DataFrames
    """
    feature_columns = [c for c in train_df.columns if c != target_column]

    X_train = train_df.select(feature_columns)
    X_test  = test_df.select(feature_columns)
    Y_train = train_df.select(target_column)
    Y_test  = test_df.select(target_column)

    logger.info(f"  Feature columns ({len(feature_columns)}): {feature_columns}")
    return X_train, X_test, Y_train, Y_test

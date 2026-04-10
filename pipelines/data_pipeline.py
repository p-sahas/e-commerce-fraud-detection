"""
Main data pipeline for e-commerce fraud detection.
Orchestrates data loading, preprocessing, and train/test splitting using PySpark.
"""

import os
import sys
import logging
from typing import Dict
from pyspark.sql import DataFrame

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from spark_session import create_spark_session, stop_spark_session
from outlier_detection import OutlierDetector, IQROutlierDetection
from preprocessing import (
    load_raw_data,
    clean_data,
    impute_missing_values,
    engineer_features,
    bin_features,
    encode_categorical_features,
    split_data,
    scale_features,
    separate_features_labels,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from spark_utils import save_dataframe, spark_to_pandas
from config import (
    get_data_paths,
    get_columns,
    get_outlier_config,
    get_binning_config,
    get_scaling_config,
)


def save_processed_data(
    X_train: DataFrame, 
    X_test: DataFrame, 
    Y_train: DataFrame, 
    Y_test: DataFrame,
    output_format: str = "both"
) -> Dict[str, str]:
    """
    Save processed data in specified format(s).
    
    Args:
        X_train, X_test, Y_train, Y_test: PySpark DataFrames
        output_format: "csv", "parquet", or "both"
        
    Returns:
        Dictionary of output paths
    """
    os.makedirs('artifacts/data', exist_ok=True)
    paths = {}
    
    if output_format in ["csv", "both"]:
        logger.info("Saving data in CSV format...")
        
        X_train_pd = spark_to_pandas(X_train)
        X_test_pd  = spark_to_pandas(X_test)
        Y_train_pd = spark_to_pandas(Y_train)
        Y_test_pd  = spark_to_pandas(Y_test)
        
        paths['X_train_csv'] = 'artifacts/data/X_train.csv'
        paths['X_test_csv']  = 'artifacts/data/X_test.csv'
        paths['Y_train_csv'] = 'artifacts/data/Y_train.csv'
        paths['Y_test_csv']  = 'artifacts/data/Y_test.csv'
        
        X_train_pd.to_csv(paths['X_train_csv'], index=False)
        X_test_pd.to_csv(paths['X_test_csv'],   index=False)
        Y_train_pd.to_csv(paths['Y_train_csv'], index=False)
        Y_test_pd.to_csv(paths['Y_test_csv'],   index=False)
        
        logger.info(" CSV files saved")
    
    if output_format in ["parquet", "both"]:
        logger.info("Saving data in Parquet format...")
        
        paths['X_train_parquet'] = 'artifacts/data/X_train.parquet'
        paths['X_test_parquet']  = 'artifacts/data/X_test.parquet'
        paths['Y_train_parquet'] = 'artifacts/data/Y_train.parquet'
        paths['Y_test_parquet']  = 'artifacts/data/Y_test.parquet'
        
        save_dataframe(X_train, paths['X_train_parquet'], format='parquet')
        save_dataframe(X_test,  paths['X_test_parquet'],  format='parquet')
        save_dataframe(Y_train, paths['Y_train_parquet'], format='parquet')
        save_dataframe(Y_test,  paths['Y_test_parquet'],  format='parquet')
        
        logger.info(" Parquet files saved")

    return paths


def data_pipeline(
        data_path: str = "data/raw/Fraud_Data.csv",
        target_column: str = "class",
        test_size: float = 0.2,
        random_state: int = get_data_paths().get('random_state', 42),
        output_format: str = "both",    
        force_rebuild: bool = False,
) -> Dict[str, str]:
    """
    Main data pipeline function to load, preprocess, and split data.
    
    Args:
        data_path: Path to the raw data file
        target_column: Name of the target column
        test_size: Proportion of data to be used as test set
        random_state: Random seed for reproducibility
        output_format: Format to save processed data ("csv", "parquet", or "both")
        force_rebuild: If True, re-run the pipeline even if outputs already exist

    Returns:
        Dictionary containing paths to saved data files
    """
    logger.info(f"\n{'='*80}")
    logger.info("STARTING PYSPARK DATA PIPELINE")
    logger.info(f"{'='*80}")

    # Short-circuit if outputs already exist and force_rebuild is False
    data_paths  = get_data_paths()
    output_check = data_paths.get('X_train', 'artifacts/data/X_train.csv')
    if not force_rebuild and os.path.exists(output_check):
        logger.info(f" Processed data already exists at '{output_check}'.")
        logger.info(" Set force_rebuild=True to regenerate. Exiting pipeline.")
        return {
            'X_train_csv': data_paths.get('X_train'),
            'X_test_csv':  data_paths.get('X_test'),
            'Y_train_csv': data_paths.get('Y_train'),
            'Y_test_csv':  data_paths.get('Y_test'),
        }

    # Input validation
    if not os.path.exists(data_path):
        logger.error(f" Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if not 0 < test_size < 1:
        logger.error(f" Invalid test_size: {test_size}")
        raise ValueError(f"test_size must be between 0 and 1, got: {test_size}")

    # Initialize Spark session
    spark = create_spark_session("FraudDetectionDataPipeline")

    try:
        # Load configurations
        columns_config  = get_columns()
        outlier_config  = get_outlier_config()
        binning_config  = get_binning_config()
        scaling_config  = get_scaling_config()

        numeric_columns  = columns_config.get("numeric_columns", [])
        nominal_columns  = columns_config.get("nominal_columns", [])
        drop_columns     = columns_config.get("drop_columns", [])
        outlier_columns  = columns_config.get("outlier_columns", [])
        outlier_method   = outlier_config.get("handling_method", "cap")
        scale_columns    = scaling_config.get("columns", [])

        # Step 1: Load raw data
        df = load_raw_data(spark, data_path)

        # Step 2: Clean data (dedup, type casting)
        df = clean_data(df, target_column, numeric_columns)

        # Step 3: Impute missing values
        df = impute_missing_values(df, numeric_columns, nominal_columns)

        # Step 4: Feature engineering (timestamps, drop configured ID columns)
        df = engineer_features(df, drop_columns)

        # Step 5: Outlier handling
        logger.info(f"\n{'='*60}")
        logger.info("STEP 5 - OUTLIER HANDLING")
        logger.info(f"{'='*60}")
        cols_to_process = [c for c in outlier_columns if c in df.columns]
        if cols_to_process:
            logger.info(f"  Processing columns: {cols_to_process} using method='{outlier_method}'")
            strategy = IQROutlierDetection(threshold=1.5, spark=None)
            detector = OutlierDetector(strategy=strategy)
            df = detector.handle_outliers(df, selected_columns=cols_to_process, method=outlier_method)
        else:
            logger.info("  No valid outlier columns found - skipping")

        # Step 6: Feature binning (replaces original column with '{col}Bins')
        df, bin_columns = bin_features(df, binning_config)

        # Step 7: Encode categorical features (nominal + bin columns)
        df, encoding_model = encode_categorical_features(
            df, nominal_columns, bin_columns, target_column
        )

        # Step 8: Split into train/test (full DataFrames, labels included)
        train_df, test_df = split_data(
            df,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state
        )

        # Step 9: Scale numeric features (fit on train, apply to both)
        # Binned columns have already replaced their originals, so only
        # unbinned numeric columns remain in scale_columns
        binned_originals = list(binning_config.keys())
        remaining_scale_cols = [c for c in scale_columns if c not in binned_originals]
        train_df, test_df = scale_features(train_df, test_df, remaining_scale_cols)

        # Step 10: Separate features and labels
        X_train, X_test, Y_train, Y_test = separate_features_labels(
            train_df, test_df, target_column
        )

        # Step 11: Save processed data
        logger.info(f"\n{'='*60}")
        logger.info("STEP 10 - SAVING PROCESSED DATA")
        logger.info(f"{'='*60}")
        output_paths = save_processed_data(X_train, X_test, Y_train, Y_test, output_format)

        logger.info(f"\n{'='*80}")
        logger.info("DATA PIPELINE COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"  Output paths: {output_paths}")

        return output_paths

    except Exception as e:
        logger.error(f" Data Pipeline Failed: {str(e)}")
        raise

    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    paths = data_pipeline()
    print("\nPipeline complete. Output files:\n")
    for key, path in paths.items():
        print(f"  {key}: {path}")
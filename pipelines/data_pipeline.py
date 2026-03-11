import os
import sys
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame, SparkSession
from pyspart.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from spark_session import create_spark_session, stop_spark_session
from spark_utils import save_dataframe, spark_to_pandas, get_dataframe_info, check_missing_values

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns


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
        # Save as CSV
        logger.info("Saving data in CSV format...")
        
        # Convert to pandas and save
        X_train_pd = spark_to_pandas(X_train)
        X_test_pd = spark_to_pandas(X_test)
        Y_train_pd = spark_to_pandas(Y_train)
        Y_test_pd = spark_to_pandas(Y_test)
        
        paths['X_train_csv'] = 'artifacts/data/X_train.csv'
        paths['X_test_csv'] = 'artifacts/data/X_test.csv'
        paths['Y_train_csv'] = 'artifacts/data/Y_train.csv'
        paths['Y_test_csv'] = 'artifacts/data/Y_test.csv'
        
        X_train_pd.to_csv(paths['X_train_csv'], index=False)
        X_test_pd.to_csv(paths['X_test_csv'], index=False)
        Y_train_pd.to_csv(paths['Y_train_csv'], index=False)
        Y_test_pd.to_csv(paths['Y_test_csv'], index=False)
        
        logger.info(" CSV files saved")
    
    if output_format in ["parquet", "both"]:
        # Save as Parquet
        logger.info("Saving data in Parquet format...")
        
        paths['X_train_parquet'] = 'artifacts/data/X_train.parquet'
        paths['X_test_parquet'] = 'artifacts/data/X_test.parquet'
        paths['Y_train_parquet'] = 'artifacts/data/Y_train.parquet'
        paths['Y_test_parquet'] = 'artifacts/data/Y_test.parquet'
        
        save_dataframe(X_train, paths['X_train_parquet'], format='parquet')
        save_dataframe(X_test, paths['X_test_parquet'], format='parquet')
        save_dataframe(Y_train, paths['Y_train_parquet'], format='parquet')
        save_dataframe(Y_test, paths['Y_test_parquet'], format='parquet')
        
        logger.info(" Parquet files saved")
    return paths


def data_pipeline(
        data_path: str = "data/raw/Fraud_Data.csv",
        target_column: str = "class",
        test_size: float = 0.2,
        random_state: int = get_data_paths('random_state'),
        output_format: str = "both",    
        force_rebuild: str  = False,

) -> Dict[str, np.ndarray]:
    """
    Main data pipeline function to load, preprocess, and split data.
    
    Args:
        data_path: Path to the raw data file
        target_column: Name of the target column
        test_size: Proportion of data to be used as test set
        random_state: Random seed for reproducibility
        output_format: Format to save processed data ("csv", "parquet", or "both")
        force_rebuild: Whether to force rebuild the data ("True", "False", "Existed")

        Returns:
        Dictionary containing paths to saved data files
        """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING PYSPARK DATA PIPELINE")
    logger.info(f"{'='*80}")
    
    # Input validation
    if not os.path.exists(data_path):
        logger.error(f" Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if not 0 < test_size < 1:
        logger.error(f" Invalid test_size: {test_size}")
        raise ValueError(f"Invalid test_size: {test_size}")
    
    # Initialize Spark session
    spark = create_spark_session("FraudDetectionDataPipeline")
    
    # Load Configurations
    try:
        data_paths = get_data_paths()
        columns = get_columns()

    except Exception as e:
        logger.error(f" Data Pipeline Filed: {str(e)}")

    finally:
        stop_spark_session(spark)
    
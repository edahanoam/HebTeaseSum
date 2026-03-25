"""Data loading utilities for Hebrew OCR text."""

from pathlib import Path
from typing import Optional

import pandas as pd


class HebrewDataLoader:
    """Handles loading and preprocessing of Hebrew OCR data.
    
    This class provides methods to load raw data files and clean them
    for downstream processing.
    """
    
    def load_raw_data(self, file_path: Path | str) -> pd.DataFrame:
        """Load and clean raw Hebrew OCR data from a CSV file.
        
        This method:
        1. Loads the CSV file with UTF-8 encoding
        2. Keeps only ['main_id', 'main_text', 'full_texts'] columns
        3. Renames them to ['id', 'summary', 'text']
        4. Removes rows where 'text' is empty
        
        Args:
            file_path: Path to the raw data CSV file.
            
        Returns:
            Cleaned DataFrame with columns: id, summary, text.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If required columns are missing or file cannot be parsed.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Load CSV with UTF-8 encoding for Hebrew support
            df = pd.read_csv(file_path, encoding="utf-8")
            
            # Validate required columns exist
            required_cols = ["main_id", "main_text", "full_texts"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns: {missing_cols}. "
                    f"Available columns: {df.columns.tolist()}"
                )
            
            # Keep only required columns and rename
            df_clean = (
                df[required_cols]
                .rename(columns={
                    "main_id": "id",
                    "main_text": "summary",
                    "full_texts": "text"
                })
            )
            
            # Remove rows where text is empty (NaN or empty string)
            df_clean = df_clean[
                df_clean["text"].notna() & (df_clean["text"].astype(str).str.strip() != "")
            ]
            
            # Reset index after filtering
            df_clean = df_clean.reset_index(drop=True)
            
            return df_clean
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading CSV file {file_path}: {e}")


class DataLoader:
    """Handles loading and preprocessing of OCR data.
    
    Attributes:
        data_dir: Path to the data directory.
    """
    
    def __init__(self, data_dir: Path | str) -> None:
        """Initialize the DataLoader.
        
        Args:
            data_dir: Path to the directory containing data files.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
    
    def load_csv(self, filename: str, encoding: str = "utf-8") -> pd.DataFrame:
        """Load a CSV file into a pandas DataFrame.
        
        Args:
            filename: Name of the CSV file to load.
            encoding: File encoding (default: utf-8 for Hebrew support).
            
        Returns:
            DataFrame containing the loaded data.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file cannot be parsed.
        """
        file_path = self.data_dir / filename
        
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading CSV file {file_path}: {e}")
    
    def save_csv(
        self, 
        df: pd.DataFrame, 
        filename: str, 
        encoding: str = "utf-8"
    ) -> None:
        """Save a DataFrame to a CSV file.
        
        Args:
            df: DataFrame to save.
            filename: Name of the output CSV file.
            encoding: File encoding (default: utf-8 for Hebrew support).
        """
        file_path = self.data_dir / filename
        
        try:
            df.to_csv(file_path, index=False, encoding=encoding)
        except Exception as e:
            raise ValueError(f"Error saving CSV file {file_path}: {e}")

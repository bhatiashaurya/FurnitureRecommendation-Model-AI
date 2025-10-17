"""
Data Preprocessing Utilities
Functions for cleaning and preparing product data for ML models
"""

import re
import ast
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class DataPreprocessor:
    @staticmethod
    def clean_price(price_str: str) -> float:
        """
        Convert price string to float
        
        Args:
            price_str: Price string like "$89.99"
            
        Returns:
            Float price value or NaN if invalid
        """
        if pd.isna(price_str) or price_str == '':
            return np.nan
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[$,]', '', str(price_str))
        
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    
    @staticmethod
    def parse_categories(categories_str: str) -> List[str]:
        """
        Parse category string to list
        
        Args:
            categories_str: String representation of list like "['Cat1', 'Cat2']"
            
        Returns:
            List of category strings
        """
        if pd.isna(categories_str) or categories_str == '':
            return []
        
        try:
            # Try to evaluate as Python literal
            return ast.literal_eval(categories_str)
        except (ValueError, SyntaxError):
            # Fallback: split by common delimiters
            return [cat.strip() for cat in re.split(r'[,;]', categories_str) if cat.strip()]
    
    @staticmethod
    def parse_images(images_str: str) -> List[str]:
        """
        Parse images string to list of URLs
        
        Args:
            images_str: String representation of image URLs
            
        Returns:
            List of image URLs
        """
        if pd.isna(images_str) or images_str == '':
            return []
        
        try:
            # Try to evaluate as Python literal
            images = ast.literal_eval(images_str)
            # Clean whitespace from URLs
            return [url.strip() for url in images if url.strip()]
        except (ValueError, SyntaxError):
            # Fallback: extract URLs
            urls = re.findall(r'https?://[^\s,\'\"]]+', images_str)
            return urls
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text for NLP processing
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def extract_main_category(categories: List[str]) -> str:
        """
        Extract main category from category list
        
        Args:
            categories: List of categories
            
        Returns:
            Main category string
        """
        if not categories:
            return "Unknown"
        
        # Return first category as main category
        return categories[0]
    
    @staticmethod
    def parse_dimensions(dim_str: str) -> Dict[str, float]:
        """
        Parse package dimensions string
        
        Args:
            dim_str: Dimension string like '24"D x 18"W x 36"H'
            
        Returns:
            Dictionary with depth, width, height
        """
        result = {'depth': np.nan, 'width': np.nan, 'height': np.nan}
        
        if pd.isna(dim_str) or dim_str == '':
            return result
        
        # Extract numbers and dimension letters
        pattern = r'([\d.]+)"?([DWH])'
        matches = re.findall(pattern, dim_str, re.IGNORECASE)
        
        for value, dim in matches:
            dim_upper = dim.upper()
            if dim_upper == 'D':
                result['depth'] = float(value)
            elif dim_upper == 'W':
                result['width'] = float(value)
            elif dim_upper == 'H':
                result['height'] = float(value)
        
        return result
    
    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess entire dataframe
        
        Args:
            df: Raw dataframe
            
        Returns:
            Preprocessed dataframe
        """
        df = df.copy()
        
        # Clean price
        if 'price' in df.columns:
            df['price_numeric'] = df['price'].apply(DataPreprocessor.clean_price)
        
        # Parse categories
        if 'categories' in df.columns:
            df['categories_list'] = df['categories'].apply(DataPreprocessor.parse_categories)
            df['main_category'] = df['categories_list'].apply(DataPreprocessor.extract_main_category)
        
        # Parse images
        if 'images' in df.columns:
            df['images_list'] = df['images'].apply(DataPreprocessor.parse_images)
            df['num_images'] = df['images_list'].apply(len)
        
        # Clean text fields
        for col in ['title', 'description']:
            if col in df.columns:
                df[f'{col}_clean'] = df[col].apply(DataPreprocessor.clean_text)
        
        # Parse dimensions
        if 'package_dimensions' in df.columns:
            dims = df['package_dimensions'].apply(DataPreprocessor.parse_dimensions)
            df['depth'] = dims.apply(lambda x: x['depth'])
            df['width'] = dims.apply(lambda x: x['width'])
            df['height'] = dims.apply(lambda x: x['height'])
        
        # Fill missing values
        df['brand'] = df['brand'].fillna('Unknown')
        df['material'] = df['material'].fillna('Mixed Materials')
        df['color'] = df['color'].fillna('Multi-Color')
        df['country_of_origin'] = df['country_of_origin'].fillna('Unknown')
        
        # Create combined text for embeddings
        df['combined_text'] = (
            df['title'].fillna('') + ' ' + 
            df['brand'].fillna('') + ' ' + 
            df['description'].fillna('') + ' ' + 
            df['main_category'].fillna('') + ' ' +
            df['material'].fillna('') + ' ' +
            df['color'].fillna('')
        )
        
        return df


# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = {
        'title': ['Modern Chair', 'Wooden Table'],
        'price': ['$89.99', '$199.50'],
        'categories': ["['Furniture', 'Chairs']", "['Furniture', 'Tables']"],
        'images': ["['http://img1.jpg', 'http://img2.jpg']", "['http://img3.jpg']"],
        'package_dimensions': ['24"D x 18"W x 36"H', '48"D x 30"W x 30"H']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df)
    
    print("\nProcessed DataFrame:")
    print(df_processed[['title', 'price_numeric', 'main_category', 'num_images']])

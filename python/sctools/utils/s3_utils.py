"""
S3 Utilities Module for Bioinformatics Workflows

This module provides a comprehensive set of tools for interacting with AWS S3 storage
within bioinformatics data processing pipelines. It facilitates reading and writing 
various data formats including AnnData objects, CSV files, 10X Genomics data, 
and other common bioinformatics file formats to and from S3 buckets.

The module is designed to integrate seamlessly with the single-cell and spatial 
analysis toolkit, providing S3 data access capabilities to all analysis modules.

Key features:
- Reading/writing AnnData objects to/from S3
- Handling 10X Genomics data format
- Processing CSV, loom, and Seurat RDS files
- Support for AWS profiles for credential management
- Directory upload/download capabilities
- Utilities for saving analysis outputs and figures to S3

Upstream dependencies:
- AWS credentials configured in ~/.aws/credentials or via environment variables
- boto3 and s3fs Python packages
- anndata package for AnnData operations

Downstream applications:
- Used by SingleCellQC, SpatialAnalysis, and other toolkit modules
- Enables cloud-based analysis workflows
- Supports persistent storage of analysis results

Author: Your Name
Date: Current Date
Version: 0.1.0
"""

import os
import boto3
import s3fs
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import h5py
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import json


class S3Utils:
    """
    Utilities for reading and writing bioinformatics data to/from AWS S3 buckets.
    
    This class provides a comprehensive interface for interacting with S3 storage
    in the context of bioinformatics data analysis. It handles various common data
    formats including AnnData objects, 10X Genomics data, CSV files, and more.
    
    The S3Utils class is designed to be used by other components in the single-cell
    and spatial analysis toolkit to enable cloud-based data storage and retrieval:
    
    - Upstream dependencies:
      * AWS credentials must be properly configured
      * boto3 and s3fs Python packages
      * File format-specific dependencies (anndata, pandas, etc.)
    
    - Downstream applications:
      * Used by SingleCellQC for loading and saving quality control results
      * Used by Normalization and other processing modules for persistent storage
      * Used by SpatialAnalysis for handling spatial data and results
      * Used by EnhancedVisualization for saving figures to S3
    
    Attributes:
        session (boto3.Session): Boto3 session object.
        s3_client (boto3.client): Boto3 S3 client.
        s3_resource (boto3.resource): Boto3 S3 resource.
        fs (s3fs.S3FileSystem): S3FileSystem object for file-like operations.
        profile_name (Optional[str]): AWS profile name.
        endpoint_url (Optional[str]): Custom S3 endpoint URL.
    
    Examples:
        >>> # Initialize with default profile
        >>> s3 = S3Utils()
        >>> 
        >>> # List available buckets
        >>> buckets = s3.list_buckets()
        >>> print(f"Available buckets: {buckets}")
        >>> 
        >>> # Read an AnnData object from S3
        >>> adata = s3.read_h5ad('my-bucket', 'path/to/file.h5ad')
        >>> 
        >>> # Save results back to S3
        >>> s3.write_h5ad(adata, 'my-bucket', 'path/to/results.h5ad')
    """
    
    def __init__(self, profile_name: Optional[str] = None, 
                 endpoint_url: Optional[str] = None):
        """
        Initialize S3Utils with optional AWS profile and endpoint.
        
        This constructor sets up the S3 connection using boto3 and s3fs, 
        with options to specify a profile from AWS credentials and/or 
        a custom endpoint URL for use with S3-compatible storage systems.
        
        Parameters:
            profile_name (Optional[str]): AWS profile name to use. If None, the default profile is used.
            endpoint_url (Optional[str]): Custom S3 endpoint URL. Use for non-AWS S3-compatible services.
        
        Examples:
            >>> # Use default AWS profile
            >>> s3 = S3Utils()
            >>> 
            >>> # Use a specific AWS profile
            >>> s3 = S3Utils(profile_name='my-profile')
            >>> 
            >>> # Connect to a custom S3-compatible service
            >>> s3 = S3Utils(endpoint_url='https://storage.googleapis.com')
            
        Notes:
            AWS credentials should be properly configured in ~/.aws/credentials
            or via environment variables before using this class.
        """
        self.profile_name = profile_name
        self.endpoint_url = endpoint_url
        
        # Initialize boto3 session with profile if provided
        session_kwargs = {}
        if profile_name:
            session_kwargs['profile_name'] = profile_name
            
        self.session = boto3.Session(**session_kwargs)
        
        # Initialize S3 client and resource
        client_kwargs = {}
        if endpoint_url:
            client_kwargs['endpoint_url'] = endpoint_url
            
        self.s3_client = self.session.client('s3', **client_kwargs)
        self.s3_resource = self.session.resource('s3', **client_kwargs)
        
        # Initialize s3fs file system
        s3fs_kwargs = {}
        if profile_name:
            s3fs_kwargs['profile'] = profile_name
        if endpoint_url:
            s3fs_kwargs['endpoint_url'] = endpoint_url
            
        self.fs = s3fs.S3FileSystem(**s3fs_kwargs)
        
        print(f"S3Utils initialized with profile: {profile_name or 'default'}")
        
    def list_buckets(self) -> List[str]:
        """
        List available S3 buckets.
        
        Retrieves a list of S3 bucket names that are accessible with 
        the current AWS credentials.
        
        Returns:
            List[str]: List of bucket names.
        
        Examples:
            >>> s3 = S3Utils()
            >>> buckets = s3.list_buckets()
            >>> print(f"Available buckets: {buckets}")
        
        Notes:
            The AWS credentials must have permission to list buckets.
        """
        response = self.s3_client.list_buckets()
        return [bucket['Name'] for bucket in response['Buckets']]
    
    def list_objects(self, bucket: str, prefix: str = '') -> List[str]:
        """
        List objects in a bucket with optional prefix.
        
        Retrieves a list of object keys in the specified S3 bucket,
        optionally filtered by a prefix.
        
        Parameters:
            bucket (str): Name of the S3 bucket.
            prefix (str): Prefix to filter objects by.
            
        Returns:
            List[str]: List of object keys.
        
        Examples:
            >>> s3 = S3Utils()
            >>> # List all objects in a bucket
            >>> objects = s3.list_objects('my-bucket')
            >>> 
            >>> # List objects with a specific prefix
            >>> objects = s3.list_objects('my-bucket', 'data/2023/')
        
        Notes:
            This method handles pagination internally, so it can list more
            than 1000 objects (the default S3 API limit).
        """
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        result = []
        for page in pages:
            if 'Contents' in page:
                result.extend([obj['Key'] for obj in page['Contents']])
                
        return result
    
    def read_h5ad(self, bucket: str, key: str, 
                  backed: bool = False,
                  tmp_dir: Optional[str] = None) -> ad.AnnData:
        """
        Read AnnData object from S3.
        
        Downloads an H5AD file from S3 and loads it into an AnnData object.
        
        Parameters:
            bucket (str): Name of the S3 bucket.
            key (str): Key (path) of the h5ad file in the bucket.
            backed (bool): Whether to load the file in backed mode (file remains on disk).
            tmp_dir (Optional[str]): Temporary directory to store the file. If None, uses '/tmp'.
            
        Returns:
            ad.AnnData: Loaded AnnData object.
        
        Examples:
            >>> s3 = S3Utils()
            >>> # Load an h5ad file from S3
            >>> adata = s3.read_h5ad('my-bucket', 'data/sample.h5ad')
            >>> 
            >>> # Load in backed mode (for large files)
            >>> adata = s3.read_h5ad('my-bucket', 'data/large_dataset.h5ad', backed=True)
        
        Notes:
            In backed mode, the file is not deleted from tmp_dir after loading.
            This is useful for large datasets that don't fit in memory.
        """
        if tmp_dir is None:
            tmp_dir = '/tmp'
            
        os.makedirs(tmp_dir, exist_ok=True)
        filename = os.path.basename(key)
        local_path = os.path.join(tmp_dir, filename)
        
        print(f"Downloading {bucket}/{key} to {local_path}...")
        self.s3_client.download_file(bucket, key, local_path)
        
        print(f"Reading {local_path} into AnnData object...")
        adata = sc.read_h5ad(local_path, backed=backed)
        
        if not backed:
            print(f"Removing temporary file {local_path}")
            os.remove(local_path)
            
        return adata
    
    def write_h5ad(self, adata: ad.AnnData, bucket: str, key: str,
                  tmp_dir: Optional[str] = None) -> None:
        """
        Write AnnData object to S3.
        
        Saves an AnnData object to a temporary file and uploads it to S3.
        
        Parameters:
            adata (ad.AnnData): AnnData object to write.
            bucket (str): Name of the S3 bucket.
            key (str): Key (path) of the h5ad file in the bucket.
            tmp_dir (Optional[str]): Temporary directory to store the file. If None, uses '/tmp'.
        
        Examples:
            >>> s3 = S3Utils()
            >>> # Save processed AnnData to S3
            >>> s3.write_h5ad(adata, 'my-bucket', 'results/processed_data.h5ad')
        
        Notes:
            The temporary file is created, uploaded to S3, and then deleted.
        """
        if tmp_dir is None:
            tmp_dir = '/tmp'
            
        os.makedirs(tmp_dir, exist_ok=True)
        filename = os.path.basename(key)
        local_path = os.path.join(tmp_dir, filename)
        
        print(f"Writing AnnData object to {local_path}...")
        adata.write(local_path)
        
        print(f"Uploading {local_path} to {bucket}/{key}...")
        self.s3_client.upload_file(local_path, bucket, key)
        
        print(f"Removing temporary file {local_path}")
        os.remove(local_path)
    
    def read_10x_mtx(self, bucket: str, prefix: str, 
                    tmp_dir: Optional[str] = None) -> ad.AnnData:
        """
        Read 10X mtx format data from S3.
        
        Downloads 10X Genomics mtx format files from S3 and loads them into an AnnData object.
        
        Parameters:
            bucket (str): Name of the S3 bucket.
            prefix (str): Prefix to the 10X mtx files directory in the bucket.
            tmp_dir (Optional[str]): Temporary directory to store the files. If None, uses '/tmp'.
            
        Returns:
            ad.AnnData: Loaded AnnData object from the 10X mtx files.
        
        Examples:
            >>> s3 = S3Utils()
            >>> # Load 10X data from S3
            >>> adata = s3.read_10x_mtx('my-bucket', 'data/sample1/')
        
        Notes:
            This method expects standard 10X Genomics output format with matrix.mtx,
            genes.tsv/features.tsv, and barcodes.tsv files.
        """
        if tmp_dir is None:
            tmp_dir = '/tmp'
            
        # Create a temporary directory for the 10X files
        temp_10x_dir = os.path.join(tmp_dir, 'temp_10x')
        os.makedirs(temp_10x_dir, exist_ok=True)
        
        # Ensure prefix ends with a slash
        if not prefix.endswith('/'):
            prefix = prefix + '/'
            
        # List all objects under the prefix
        objects = self.list_objects(bucket, prefix)
        
        # Download each file
        for obj_key in objects:
            # Skip directory objects
            if obj_key.endswith('/'):
                continue
                
            # Get filename without the prefix
            rel_path = obj_key[len(prefix):]
            local_file_path = os.path.join(temp_10x_dir, rel_path)
            
            # Create directories if needed
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            print(f"Downloading {bucket}/{obj_key} to {local_file_path}...")
            self.s3_client.download_file(bucket, obj_key, local_file_path)
        
        print(f"Reading 10X data from {temp_10x_dir}...")
        adata = sc.read_10x_mtx(temp_10x_dir)
        
        print(f"Removing temporary directory {temp_10x_dir}")
        import shutil
        shutil.rmtree(temp_10x_dir)
        
        return adata
    
    def read_csv(self, bucket: str, key: str, **kwargs) -> pd.DataFrame:
        """
        Read CSV file from S3 into a pandas DataFrame.
        
        Opens a CSV file directly from S3 and reads it into a pandas DataFrame.
        
        Parameters:
            bucket (str): Name of the S3 bucket.
            key (str): Key (path) of the CSV file in the bucket.
            **kwargs: Additional arguments to pass to pd.read_csv.
            
        Returns:
            pd.DataFrame: Loaded pandas DataFrame.
        
        Examples:
            >>> s3 = S3Utils()
            >>> # Read a CSV file with default parameters
            >>> df = s3.read_csv('my-bucket', 'data/metadata.csv')
            >>> 
            >>> # Read with custom parameters
            >>> df = s3.read_csv('my-bucket', 'data/counts.csv', 
            ...                index_col=0, sep='\t')
        
        Notes:
            This method uses s3fs to read the file directly without downloading it first.
            All pandas.read_csv parameters are supported through **kwargs.
        """
        s3_path = f"s3://{bucket}/{key}"
        print(f"Reading CSV from {s3_path}...")
        
        with self.fs.open(s3_path, 'rb') as f:
            df = pd.read_csv(f, **kwargs)
            
        return df
    
    def write_csv(self, df: pd.DataFrame, bucket: str, key: str, **kwargs) -> None:
        """
        Write pandas DataFrame to S3 as a CSV file.
        
        Saves a pandas DataFrame directly to S3 as a CSV file.
        
        Parameters:
            df (pd.DataFrame): DataFrame to write.
            bucket (str): Name of the S3 bucket.
            key (str): Key (path) of the CSV file in the bucket.
            **kwargs: Additional arguments to pass to df.to_csv.
        
        Examples:
            >>> s3 = S3Utils()
            >>> # Write a DataFrame to CSV with default parameters
            >>> s3.write_csv(results_df, 'my-bucket', 'results/stats.csv')
            >>> 
            >>> # Write with custom parameters
            >>> s3.write_csv(results_df, 'my-bucket', 'results/stats.csv',
            ...             index=False, float_format='%.3f')
        
        Notes:
            This method uses s3fs to write the file directly without creating a local copy first.
            All pandas.DataFrame.to_csv parameters are supported through **kwargs.
        """
        s3_path = f"s3://{bucket}/{key}"
        print(f"Writing DataFrame to {s3_path}...")
        
        with self.fs.open(s3_path, 'w') as f:
            df.to_csv(f, **kwargs)
    
    def read_parquet(self, bucket: str, key: str, **kwargs) -> pd.DataFrame:
        """
        Read Parquet file from S3 into a pandas DataFrame.
        
        Opens a Parquet file directly from S3 and reads it into a pandas DataFrame.
        
        Parameters:
            bucket (str): Name of the S3 bucket.
            key (str): Key (path) of the Parquet file in the bucket.
            **kwargs: Additional arguments to pass to pd.read_parquet.
            
        Returns:
            pd.DataFrame: Loaded pandas DataFrame.
            
        Examples:
            >>> s3 = S3Utils()
            >>> # Read a Parquet file
            >>> df = s3.read_parquet('my-bucket', 'data/matrix.parquet')
        
        Notes:
            This method requires the pyarrow or fastparquet package.
            All pandas.read_parquet parameters are supported through **kwargs.
        """
        s3_path = f"s3://{bucket}/{key}"
        print(f"Reading Parquet from {s3_path}...")
        
        with self.fs.open(s3_path, 'rb') as f:
            df = pd.read_parquet(f, **kwargs)
            
        return df
    
    def write_parquet(self, df: pd.DataFrame, bucket: str, key: str, **kwargs) -> None:
        """
        Write pandas DataFrame to S3 as a Parquet file.
        
        Saves a pandas DataFrame directly to S3 as a Parquet file.
        
        Parameters:
            df (pd.DataFrame): DataFrame to write.
            bucket (str): Name of the S3 bucket.
            key (str): Key (path) of the Parquet file in the bucket.
            **kwargs: Additional arguments to pass to df.to_parquet.
            
        Examples:
            >>> s3 = S3Utils()
            >>> # Write a DataFrame to Parquet
            >>> s3.write_parquet(results_df, 'my-bucket', 'results/matrix.parquet')
        
        Notes:
            This method requires the pyarrow or fastparquet package.
            All pandas.DataFrame.to_parquet parameters are supported through **kwargs.
        """
        s3_path = f"s3://{bucket}/{key}"
        print(f"Writing DataFrame to {s3_path} as Parquet...")
        
        with self.fs.open(s3_path, 'wb') as f:
            df.to_parquet(f, **kwargs)
    
    def read_zarr(self, bucket: str, key: str, **kwargs) -> ad.AnnData:
        """
        Read Zarr store from S3 into an AnnData object.
        
        Loads a Zarr store from S3 into an AnnData object.
        
        Parameters:
            bucket (str): Name of the S3 bucket.
            key (str): Key (path) of the Zarr store in the bucket.
            **kwargs: Additional arguments to pass to ad.read_zarr.
            
        Returns:
            ad.AnnData: Loaded AnnData object.
            
        Examples:
            >>> s3 = S3Utils()
            >>> # Read a Zarr store
            >>> adata = s3.read_zarr('my-bucket', 'data/anndata.zarr')
        
        Notes:
            This method requires the zarr package.
            The key should point to the root of the Zarr store (directory).
        """
        s3_path = f"s3://{bucket}/{key}"
        print(f"Reading Zarr store from {s3_path}...")
        
        # Zarr requires a store interface, which we create via s3fs
        store = s3fs.S3Map(root=s3_path, s3=self.fs)
        
        # Read the AnnData object from the Zarr store
        adata = ad.read_zarr(store, **kwargs)
        
        return adata
    
    def write_zarr(self, adata: ad.AnnData, bucket: str, key: str, **kwargs) -> None:
        """
        Write AnnData object to S3 as a Zarr store.
        
        Saves an AnnData object directly to S3 as a Zarr store.
        
        Parameters:
            adata (ad.AnnData): AnnData object to write.
            bucket (str): Name of the S3 bucket.
            key (str): Key (path) of the Zarr store in the bucket.
            **kwargs: Additional arguments to pass to adata.write_zarr.
            
        Examples:
            >>> s3 = S3Utils()
            >>> # Write an AnnData object to Zarr
            >>> s3.write_zarr(adata, 'my-bucket', 'results/anndata.zarr')
        
        Notes:
            This method requires the zarr package.
            The key should point to the desired location of the Zarr store root.
            Zarr stores are directory-like structures, so the key should typically end with '.zarr'.
        """
        s3_path = f"s3://{bucket}/{key}"
        print(f"Writing AnnData object to {s3_path} as Zarr store...")
        
        # Create an S3 store interface via s3fs
        store = s3fs.S3Map(root=s3_path, s3=self.fs)
        
        # Write the AnnData object to the Zarr store
        adata.write_zarr(store, **kwargs)
    
    def save_figure(self, fig, bucket: str, key: str, 
                   format: str = 'png', **kwargs) -> None:
        """
        Save a matplotlib figure to S3.
        
        Renders a matplotlib figure to a buffer and uploads it to S3.
        
        Parameters:
            fig (matplotlib.figure.Figure): Figure to save.
            bucket (str): Name of the S3 bucket.
            key (str): Key (path) of the file in the bucket.
            format (str): File format (e.g., 'png', 'pdf', 'svg').
            **kwargs: Additional arguments to pass to fig.savefig.
        
        Examples:
            >>> import matplotlib.pyplot as plt
            >>> s3 = S3Utils()
            >>> 
            >>> # Create a figure
            >>> fig, ax = plt.subplots()
            >>> ax.plot([1, 2, 3], [4, 5, 6])
            >>> 
            >>> # Save figure to S3
            >>> s3.save_figure(fig, 'my-bucket', 'results/plot.png', dpi=300)
        
        Notes:
            This method supports all matplotlib savefig formats and parameters.
            The Content-Type of the S3 object is set based on the format.
        """
        import io
        import matplotlib.pyplot as plt
        
        # Create a bytes buffer
        buf = io.BytesIO()
        
        # Save the figure to the buffer
        fig.savefig(buf, format=format, **kwargs)
        buf.seek(0)
        
        # Determine content type based on format
        content_type_map = {
            'png': 'image/png',
            'pdf': 'application/pdf',
            'svg': 'image/svg+xml',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg'
        }
        content_type = content_type_map.get(format, 'application/octet-stream')
        
        # Upload the buffer to S3
        self.s3_client.put_object(
            Body=buf.getvalue(),
            Bucket=bucket,
            Key=key,
            ContentType=content_type
        )
        
        print(f"Saved figure to {bucket}/{key}")
        
    def upload_directory(self, local_dir: str, bucket: str, prefix: str) -> None:
        """
        Upload a local directory to S3.
        
        Recursively uploads all files from a local directory to an S3 bucket,
        preserving the directory structure.
        
        Parameters:
            local_dir (str): Path to the local directory.
            bucket (str): Name of the S3 bucket.
            prefix (str): Prefix (path) in the bucket to upload to.
        
        Examples:
            >>> s3 = S3Utils()
            >>> # Upload an entire results directory to S3
            >>> s3.upload_directory('/path/to/results', 'my-bucket', 'project/results')
        
        Notes:
            This method preserves the directory structure relative to local_dir.
            For example, if local_dir contains a file 'subdir/file.txt', it will
            be uploaded to '{prefix}/subdir/file.txt' in the bucket.
        """
        # Ensure prefix ends with a slash if it's not empty
        if prefix and not prefix.endswith('/'):
            prefix = prefix + '/'
            
        # Walk through the directory
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                # Get the full local path
                local_path = os.path.join(root, file)
                
                # Get the relative path from the source directory
                rel_path = os.path.relpath(local_path, local_dir)
                
                # Calculate the S3 key
                s3_key = prefix + rel_path
                
                print(f"Uploading {local_path} to {bucket}/{s3_key}...")
                self.s3_client.upload_file(local_path, bucket, s3_key)
    
    def download_directory(self, bucket: str, prefix: str, local_dir: str) -> None:
        """
        Download a directory from S3 to a local path.
        
        Recursively downloads all files from an S3 bucket with a given prefix
        to a local directory, preserving the directory structure.
        
        Parameters:
            bucket (str): Name of the S3 bucket.
            prefix (str): Prefix (path) in the bucket to download from.
            local_dir (str): Path to the local directory.
        
        Examples:
            >>> s3 = S3Utils()
            >>> # Download all files from a project in S3
            >>> s3.download_directory('my-bucket', 'project/', '/path/to/local/project')
        
        Notes:
            This method preserves the directory structure relative to the prefix.
            For example, if the bucket contains '{prefix}/subdir/file.txt', it will
            be downloaded to '{local_dir}/subdir/file.txt'.
        """
        # Ensure prefix ends with a slash if it's not empty
        if prefix and not prefix.endswith('/'):
            prefix = prefix + '/'
            
        # Create the local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # List all objects under the prefix
        objects = self.list_objects(bucket, prefix)
        
        for obj_key in objects:
            # Skip directory objects
            if obj_key.endswith('/'):
                continue
                
            # Remove the prefix from the object key to get the relative path
            rel_path = obj_key[len(prefix):] if obj_key.startswith(prefix) else obj_key
            
            # Calculate the local path
            local_path = os.path.join(local_dir, rel_path)
            
            # Create parent directories if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            print(f"Downloading {bucket}/{obj_key} to {local_path}...")
            self.s3_client.download_file(bucket, obj_key, local_path)

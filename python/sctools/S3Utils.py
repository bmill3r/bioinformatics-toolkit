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
    Utilities for reading and writing data to/from AWS S3 buckets.
    """
    
    def __init__(self, profile_name: Optional[str] = None, 
                 endpoint_url: Optional[str] = None):
        """
        Initialize S3Utils with optional AWS profile and endpoint.
        
        Parameters:
        -----------
        profile_name : Optional[str]
            AWS profile name to use. If None, the default profile is used.
        endpoint_url : Optional[str]
            Custom S3 endpoint URL. Use for non-AWS S3-compatible services.
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
        
        Returns:
        --------
        List[str]
            List of bucket names.
        """
        response = self.s3_client.list_buckets()
        return [bucket['Name'] for bucket in response['Buckets']]
    
    def list_objects(self, bucket: str, prefix: str = '') -> List[str]:
        """
        List objects in a bucket with optional prefix.
        
        Parameters:
        -----------
        bucket : str
            Name of the S3 bucket.
        prefix : str
            Prefix to filter objects by.
            
        Returns:
        --------
        List[str]
            List of object keys.
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
        
        Parameters:
        -----------
        bucket : str
            Name of the S3 bucket.
        key : str
            Key (path) of the h5ad file in the bucket.
        backed : bool
            Whether to load the file in backed mode (file remains on disk).
        tmp_dir : Optional[str]
            Temporary directory to store the file. If None, uses '/tmp'.
            
        Returns:
        --------
        ad.AnnData
            Loaded AnnData object.
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
        
        Parameters:
        -----------
        adata : ad.AnnData
            AnnData object to write.
        bucket : str
            Name of the S3 bucket.
        key : str
            Key (path) of the h5ad file in the bucket.
        tmp_dir : Optional[str]
            Temporary directory to store the file. If None, uses '/tmp'.
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
        
        Parameters:
        -----------
        bucket : str
            Name of the S3 bucket.
        prefix : str
            Prefix to the 10X mtx files directory in the bucket.
        tmp_dir : Optional[str]
            Temporary directory to store the files. If None, uses '/tmp'.
            
        Returns:
        --------
        ad.AnnData
            Loaded AnnData object from the 10X mtx files.
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
        
        Parameters:
        -----------
        bucket : str
            Name of the S3 bucket.
        key : str
            Key (path) of the CSV file in the bucket.
        **kwargs :
            Additional arguments to pass to pd.read_csv.
            
        Returns:
        --------
        pd.DataFrame
            Loaded pandas DataFrame.
        """
        s3_path = f"s3://{bucket}/{key}"
        print(f"Reading CSV from {s3_path}...")
        
        with self.fs.open(s3_path, 'rb') as f:
            df = pd.read_csv(f, **kwargs)
            
        return df
    
    def write_csv(self, df: pd.DataFrame, bucket: str, key: str, **kwargs) -> None:
        """
        Write pandas DataFrame to S3 as a CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to write.
        bucket : str
            Name of the S3 bucket.
        key : str
            Key (path) of the CSV file in the bucket.
        **kwargs :
            Additional arguments to pass to df.to_csv.
        """
        s3_path = f"s3://{bucket}/{key}"
        print(f"Writing DataFrame to {s3_path}...")
        
        with self.fs.open(s3_path, 'w') as f:
            df.to_csv(f, **kwargs)
    
    def read_loom(self, bucket: str, key: str, 
                 tmp_dir: Optional[str] = None) -> ad.AnnData:
        """
        Read loom file from S3 into AnnData.
        
        Parameters:
        -----------
        bucket : str
            Name of the S3 bucket.
        key : str
            Key (path) of the loom file in the bucket.
        tmp_dir : Optional[str]
            Temporary directory to store the file. If None, uses '/tmp'.
            
        Returns:
        --------
        ad.AnnData
            Loaded AnnData object.
        """
        if tmp_dir is None:
            tmp_dir = '/tmp'
            
        os.makedirs(tmp_dir, exist_ok=True)
        filename = os.path.basename(key)
        local_path = os.path.join(tmp_dir, filename)
        
        print(f"Downloading {bucket}/{key} to {local_path}...")
        self.s3_client.download_file(bucket, key, local_path)
        
        print(f"Reading {local_path} into AnnData object...")
        adata = sc.read_loom(local_path)
        
        print(f"Removing temporary file {local_path}")
        os.remove(local_path)
            
        return adata
    
    def read_seurat(self, bucket: str, key: str, 
                  tmp_dir: Optional[str] = None) -> ad.AnnData:
        """
        Read Seurat RDS file from S3 into AnnData using anndata2ri.
        
        Parameters:
        -----------
        bucket : str
            Name of the S3 bucket.
        key : str
            Key (path) of the RDS file in the bucket.
        tmp_dir : Optional[str]
            Temporary directory to store the file. If None, uses '/tmp'.
            
        Returns:
        --------
        ad.AnnData
            Loaded AnnData object.
        """
        try:
            import anndata2ri
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr
        except ImportError:
            raise ImportError("This function requires the packages anndata2ri and rpy2. "
                             "Please install them with: pip install anndata2ri rpy2")
            
        if tmp_dir is None:
            tmp_dir = '/tmp'
            
        os.makedirs(tmp_dir, exist_ok=True)
        filename = os.path.basename(key)
        local_path = os.path.join(tmp_dir, filename)
        
        print(f"Downloading {bucket}/{key} to {local_path}...")
        self.s3_client.download_file(bucket, key, local_path)
        
        print(f"Reading {local_path} into AnnData object via Seurat conversion...")
        
        # Activate conversion between anndata and R objects
        anndata2ri.activate()
        
        # Import R packages
        base = importr('base')
        seurat = importr('Seurat')
        
        # Load the Seurat object in R
        r_seurat = base.readRDS(local_path)
        
        # Convert Seurat to SingleCellExperiment
        sce = seurat.as_SingleCellExperiment(r_seurat)
        
        # Convert to AnnData
        adata = anndata2ri.rpy2py(sce)
        
        # Deactivate conversion
        anndata2ri.deactivate()
        
        print(f"Removing temporary file {local_path}")
        os.remove(local_path)
            
        return adata
    
    def upload_directory(self, local_dir: str, bucket: str, prefix: str) -> None:
        """
        Upload a local directory to S3.
        
        Parameters:
        -----------
        local_dir : str
            Path to the local directory.
        bucket : str
            Name of the S3 bucket.
        prefix : str
            Prefix (path) in the bucket to upload to.
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
        
        Parameters:
        -----------
        bucket : str
            Name of the S3 bucket.
        prefix : str
            Prefix (path) in the bucket to download from.
        local_dir : str
            Path to the local directory.
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
    
    def read_file(self, bucket: str, key: str) -> bytes:
        """
        Read a file from S3 into memory.
        
        Parameters:
        -----------
        bucket : str
            Name of the S3 bucket.
        key : str
            Key (path) of the file in the bucket.
            
        Returns:
        --------
        bytes
            File content as bytes.
        """
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    
    def write_file(self, content: Union[bytes, str], bucket: str, key: str) -> None:
        """
        Write content to a file in S3.
        
        Parameters:
        -----------
        content : Union[bytes, str]
            Content to write to the file.
        bucket : str
            Name of the S3 bucket.
        key : str
            Key (path) of the file in the bucket.
        """
        # Convert string to bytes if needed
        if isinstance(content, str):
            content = content.encode('utf-8')
            
        self.s3_client.put_object(Body=content, Bucket=bucket, Key=key)
        print(f"Wrote content to {bucket}/{key}")
    
    def save_figure(self, fig, bucket: str, key: str, 
                   format: str = 'png', **kwargs) -> None:
        """
        Save a matplotlib figure to S3.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to save.
        bucket : str
            Name of the S3 bucket.
        key : str
            Key (path) of the file in the bucket.
        format : str
            File format (e.g., 'png', 'pdf', 'svg').
        **kwargs :
            Additional arguments to pass to fig.savefig.
        """
        import io
        import matplotlib.pyplot as plt
        
        # Create a bytes buffer
        buf = io.BytesIO()
        
        # Save the figure to the buffer
        fig.savefig(buf, format=format, **kwargs)
        buf.seek(0)
        
        # Upload the buffer to S3
        self.s3_client.put_object(
            Body=buf.getvalue(),
            Bucket=bucket,
            Key=key,
            ContentType=f'image/{format}'
        )
        
        print(f"Saved figure to {bucket}/{key}")
        
# Example usage
if __name__ == "__main__":
    # Initialize with default profile
    s3utils = S3Utils()
    
    # List available buckets
    buckets = s3utils.list_buckets()
    print(f"Available buckets: {buckets}")
    
    # Using a different profile
    s3utils_prof = S3Utils(profile_name='my-profile')
    
    # Example reading an h5ad file
    # adata = s3utils.read_h5ad('my-bucket', 'path/to/file.h5ad')
    
    # Example writing an h5ad file
    # s3utils.write_h5ad(adata, 'my-bucket', 'path/to/output.h5ad')

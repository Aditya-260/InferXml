"""
MinIO Storage Service
Handles all object storage operations for datasets, models, and artifacts
"""
import os
import io
import json
from typing import Optional, BinaryIO, Dict, Any
from datetime import timedelta
from minio import Minio
from minio.error import S3Error


class MinIOService:
    """Service for interacting with MinIO object storage"""
    
    # Bucket names
    BUCKET_DATASETS = 'datasets'
    BUCKET_MODELS = 'models'
    BUCKET_ARTIFACTS = 'artifacts'
    
    def __init__(
        self,
        endpoint: str = None,
        access_key: str = None,
        secret_key: str = None,
        secure: bool = False
    ):
        """
        Initialize MinIO client
        
        Args:
            endpoint: MinIO server endpoint (e.g., 'localhost:9000')
            access_key: Access key for authentication
            secret_key: Secret key for authentication
            secure: Use HTTPS if True
        """
        self.endpoint = endpoint or os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        self.access_key = access_key or os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.secret_key = secret_key or os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        self.secure = secure or os.getenv('MINIO_SECURE', 'false').lower() == 'true'
        
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        
        # Ensure buckets exist
        self._ensure_buckets()
    
    def _ensure_buckets(self):
        """Create required buckets if they don't exist"""
        buckets = [self.BUCKET_DATASETS, self.BUCKET_MODELS, self.BUCKET_ARTIFACTS]
        
        for bucket in buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    print(f"Created bucket: {bucket}")
            except S3Error as e:
                print(f"Error creating bucket {bucket}: {e}")
    
    # ==================== UPLOAD OPERATIONS ====================
    
    def upload_file(
        self,
        bucket: str,
        object_name: str,
        file_path: str,
        content_type: str = None
    ) -> bool:
        """
        Upload a file from disk
        
        Args:
            bucket: Target bucket name
            object_name: Object name in bucket
            file_path: Local path to file
            content_type: MIME type of file
            
        Returns:
            True if successful
        """
        try:
            self.client.fput_object(
                bucket,
                object_name,
                file_path,
                content_type=content_type
            )
            return True
        except S3Error as e:
            print(f"Error uploading file: {e}")
            return False
    
    def upload_bytes(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str = 'application/octet-stream'
    ) -> bool:
        """
        Upload bytes directly
        
        Args:
            bucket: Target bucket name
            object_name: Object name in bucket
            data: Bytes to upload
            content_type: MIME type
            
        Returns:
            True if successful
        """
        try:
            print(f"🔍 MinIO: Uploading {len(data)} bytes to '{bucket}/{object_name}'", flush=True)
            data_stream = io.BytesIO(data)
            self.client.put_object(
                bucket,
                object_name,
                data_stream,
                length=len(data),
                content_type=content_type
            )
            print(f"✅ MinIO: Upload successful", flush=True)
            return True
        except S3Error as e:
            print(f"❌ MinIO Error uploading bytes: {e}", flush=True)
            return False
    
    def upload_stream(
        self,
        bucket: str,
        object_name: str,
        stream: BinaryIO,
        length: int,
        content_type: str = 'application/octet-stream'
    ) -> bool:
        """
        Upload from a file stream
        
        Args:
            bucket: Target bucket name
            object_name: Object name in bucket
            stream: File-like object
            length: Size of data
            content_type: MIME type
            
        Returns:
            True if successful
        """
        try:
            self.client.put_object(
                bucket,
                object_name,
                stream,
                length=length,
                content_type=content_type
            )
            return True
        except S3Error as e:
            print(f"Error uploading stream: {e}")
            return False
    
    def upload_json(
        self,
        bucket: str,
        object_name: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Upload JSON data
        
        Args:
            bucket: Target bucket name
            object_name: Object name in bucket
            data: Dictionary to serialize as JSON
            
        Returns:
            True if successful
        """
        json_bytes = json.dumps(data, indent=2).encode('utf-8')
        return self.upload_bytes(bucket, object_name, json_bytes, 'application/json')
    
    # ==================== DOWNLOAD OPERATIONS ====================
    
    def download_file(
        self,
        bucket: str,
        object_name: str,
        file_path: str
    ) -> bool:
        """
        Download object to file
        
        Args:
            bucket: Source bucket name
            object_name: Object name in bucket
            file_path: Local path to save file
            
        Returns:
            True if successful
        """
        try:
            self.client.fget_object(bucket, object_name, file_path)
            return True
        except S3Error as e:
            print(f"Error downloading file: {e}")
            return False
    
    def download_bytes(
        self,
        bucket: str,
        object_name: str
    ) -> Optional[bytes]:
        """
        Download object as bytes
        
        Args:
            bucket: Source bucket name
            object_name: Object name in bucket
            
        Returns:
            Bytes content or None if failed
        """
        try:
            print(f"🔍 MinIO: Getting object '{object_name}' from bucket '{bucket}'", flush=True)
            response = self.client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            print(f"✅ MinIO: Successfully downloaded {len(data)} bytes", flush=True)
            return data
        except S3Error as e:
            print(f"❌ MinIO Error downloading '{object_name}' from '{bucket}': {e}", flush=True)
            return None
    
    def download_json(
        self,
        bucket: str,
        object_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Download and parse JSON object
        
        Args:
            bucket: Source bucket name
            object_name: Object name in bucket
            
        Returns:
            Parsed dictionary or None if failed
        """
        data = self.download_bytes(bucket, object_name)
        if data:
            return json.loads(data.decode('utf-8'))
        return None
    
    # ==================== URL OPERATIONS ====================
    
    def get_presigned_url(
        self,
        bucket: str,
        object_name: str,
        expires: timedelta = timedelta(hours=1)
    ) -> Optional[str]:
        """
        Generate presigned URL for downloading
        
        Args:
            bucket: Bucket name
            object_name: Object name
            expires: URL expiration time
            
        Returns:
            Presigned URL or None if failed
        """
        try:
            return self.client.presigned_get_object(bucket, object_name, expires=expires)
        except S3Error as e:
            print(f"Error generating presigned URL: {e}")
            return None
    
    def get_upload_url(
        self,
        bucket: str,
        object_name: str,
        expires: timedelta = timedelta(hours=1)
    ) -> Optional[str]:
        """
        Generate presigned URL for uploading
        
        Args:
            bucket: Bucket name
            object_name: Object name
            expires: URL expiration time
            
        Returns:
            Presigned URL or None if failed
        """
        try:
            return self.client.presigned_put_object(bucket, object_name, expires=expires)
        except S3Error as e:
            print(f"Error generating upload URL: {e}")
            return None
    
    # ==================== DELETE OPERATIONS ====================
    
    def delete_object(self, bucket: str, object_name: str) -> bool:
        """Delete an object"""
        try:
            self.client.remove_object(bucket, object_name)
            return True
        except S3Error as e:
            print(f"Error deleting object: {e}")
            return False
    
    def delete_objects(self, bucket: str, prefix: str) -> bool:
        """Delete all objects with given prefix"""
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            for obj in objects:
                self.client.remove_object(bucket, obj.object_name)
            return True
        except S3Error as e:
            print(f"Error deleting objects: {e}")
            return False
    
    # ==================== LIST OPERATIONS ====================
    
    def list_objects(
        self,
        bucket: str,
        prefix: str = '',
        recursive: bool = False
    ) -> list:
        """
        List objects in bucket
        
        Args:
            bucket: Bucket name
            prefix: Filter by prefix
            recursive: Include nested objects
            
        Returns:
            List of object info dictionaries
        """
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=recursive)
            return [
                {
                    'name': obj.object_name,
                    'size': obj.size,
                    'modified': obj.last_modified,
                    'etag': obj.etag
                }
                for obj in objects
            ]
        except S3Error as e:
            print(f"Error listing objects: {e}")
            return []
    
    def object_exists(self, bucket: str, object_name: str) -> bool:
        """Check if object exists"""
        try:
            self.client.stat_object(bucket, object_name)
            return True
        except S3Error:
            return False
    
    # ==================== DATASET HELPERS ====================
    
    def upload_dataset(
        self,
        user_id: int,
        dataset_id: int,
        filename: str,
        file_stream: BinaryIO,
        file_size: int,
        content_type: str
    ) -> str:
        """
        Upload a dataset file
        
        Returns:
            Object path in MinIO
        """
        object_name = f"user_{user_id}/dataset_{dataset_id}/{filename}"
        success = self.upload_stream(
            self.BUCKET_DATASETS,
            object_name,
            file_stream,
            file_size,
            content_type
        )
        return object_name if success else None
    
    def get_dataset_url(self, object_path: str) -> Optional[str]:
        """Get download URL for dataset"""
        return self.get_presigned_url(self.BUCKET_DATASETS, object_path)
    
    # ==================== MODEL HELPERS ====================
    
    def upload_model_package(
        self,
        user_id: int,
        experiment_id: int,
        package_dir: str
    ) -> str:
        """
        Upload entire model package directory
        
        Args:
            user_id: User ID
            experiment_id: Experiment ID
            package_dir: Local directory containing model package
            
        Returns:
            Base path in MinIO
        """
        base_path = f"user_{user_id}/experiment_{experiment_id}"
        
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, package_dir)
                object_name = f"{base_path}/{relative_path}"
                
                self.upload_file(self.BUCKET_MODELS, object_name, local_path)
        
        return base_path
    
    def download_model_package(
        self,
        model_path: str,
        local_dir: str
    ) -> bool:
        """
        Download model package to local directory
        
        Args:
            model_path: Base path in MinIO
            local_dir: Local directory to save files
            
        Returns:
            True if successful
        """
        os.makedirs(local_dir, exist_ok=True)
        
        objects = self.list_objects(self.BUCKET_MODELS, prefix=model_path, recursive=True)
        
        for obj in objects:
            relative_path = obj['name'].replace(model_path + '/', '')
            local_path = os.path.join(local_dir, relative_path)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.download_file(self.BUCKET_MODELS, obj['name'], local_path)
        
        return True


# Singleton instance
_minio_service = None


def get_minio_service() -> MinIOService:
    """Get or create MinIO service instance"""
    global _minio_service
    if _minio_service is None:
        _minio_service = MinIOService()
    return _minio_service

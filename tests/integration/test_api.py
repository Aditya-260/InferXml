"""
Integration Tests for API Routes
"""
import pytest
import json
from io import BytesIO


@pytest.fixture
def app():
    """Create test application"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))
    
    from app import create_app, db
    
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def auth_headers(client):
    """Create authenticated user and return headers"""
    # Register user
    client.post('/api/auth/register', json={
        'email': 'test@example.com',
        'username': 'testuser',
        'password': 'testpass123'
    })
    
    # Login
    response = client.post('/api/auth/login', json={
        'email': 'test@example.com',
        'password': 'testpass123'
    })
    
    data = json.loads(response.data)
    token = data.get('access_token')
    
    return {'Authorization': f'Bearer {token}'}


class TestAuthRoutes:
    """Test authentication routes"""
    
    def test_register(self, client):
        """Test user registration"""
        response = client.post('/api/auth/register', json={
            'email': 'newuser@example.com',
            'username': 'newuser',
            'password': 'password123'
        })
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert 'user' in data
    
    def test_register_duplicate_email(self, client):
        """Test duplicate email rejection"""
        # First registration
        client.post('/api/auth/register', json={
            'email': 'duplicate@example.com',
            'username': 'user1',
            'password': 'password123'
        })
        
        # Duplicate attempt
        response = client.post('/api/auth/register', json={
            'email': 'duplicate@example.com',
            'username': 'user2',
            'password': 'password123'
        })
        
        assert response.status_code == 400
    
    def test_login(self, client):
        """Test user login"""
        # Register first
        client.post('/api/auth/register', json={
            'email': 'login@example.com',
            'username': 'loginuser',
            'password': 'password123'
        })
        
        # Login
        response = client.post('/api/auth/login', json={
            'email': 'login@example.com',
            'password': 'password123'
        })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'access_token' in data
    
    def test_login_wrong_password(self, client):
        """Test login with wrong password"""
        # Register first
        client.post('/api/auth/register', json={
            'email': 'wrong@example.com',
            'username': 'wronguser',
            'password': 'correctpassword'
        })
        
        # Login with wrong password
        response = client.post('/api/auth/login', json={
            'email': 'wrong@example.com',
            'password': 'wrongpassword'
        })
        
        assert response.status_code == 401
    
    def test_me_route(self, client, auth_headers):
        """Test get current user route"""
        response = client.get('/api/auth/me', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['user']['email'] == 'test@example.com'


class TestDatasetRoutes:
    """Test dataset management routes"""
    
    def test_list_datasets_empty(self, client, auth_headers):
        """Test listing empty datasets"""
        response = client.get('/api/datasets', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['datasets'] == []
    
    def test_upload_dataset_no_file(self, client, auth_headers):
        """Test upload without file"""
        response = client.post(
            '/api/datasets/upload',
            headers=auth_headers,
            data={}
        )
        
        assert response.status_code == 400


class TestModelRoutes:
    """Test model management routes"""
    
    def test_list_models_empty(self, client, auth_headers):
        """Test listing empty models"""
        response = client.get('/api/models', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['models'] == []
    
    def test_get_nonexistent_model(self, client, auth_headers):
        """Test getting non-existent model"""
        response = client.get('/api/models/999', headers=auth_headers)
        
        assert response.status_code == 404


class TestProtectedRoutes:
    """Test route protection"""
    
    def test_unauthorized_access(self, client):
        """Test accessing protected route without token"""
        response = client.get('/api/datasets')
        
        assert response.status_code == 401
    
    def test_invalid_token(self, client):
        """Test accessing with invalid token"""
        headers = {'Authorization': 'Bearer invalid_token'}
        response = client.get('/api/datasets', headers=headers)
        
        assert response.status_code == 422  # Unprocessable token

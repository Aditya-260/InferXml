"""
Database Migrations Utility
Initialize and manage database schema
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from app import create_app, db
from app.models import User, Dataset, Experiment, TrainingJob


def init_db():
    """Initialize database with all tables"""
    app = create_app()
    
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully!")
        
        # Print created tables
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        print(f"\n📦 Tables created: {len(tables)}")
        for table in tables:
            print(f"   - {table}")


def drop_db():
    """Drop all database tables"""
    app = create_app()
    
    with app.app_context():
        db.drop_all()
        print("⚠️ All database tables dropped!")


def seed_db():
    """Seed database with sample data"""
    app = create_app()
    
    with app.app_context():
        # Check if user already exists
        existing_user = User.query.filter_by(email='demo@inferx.ml').first()
        if existing_user:
            print("Demo user already exists. Skipping seed.")
            return
        
        # Create demo user
        demo_user = User(
            email='demo@inferx.ml',
            username='demo'
        )
        demo_user.set_password('demo123')
        db.session.add(demo_user)
        
        # Create sample dataset metadata
        sample_dataset = Dataset(
            name='Sample Customer Churn',
            file_type='csv',
            file_size=102400,
            file_path='demo/sample_churn.csv',
            data_type='tabular',
            num_rows=1000,
            num_columns=10,
            profile_status='completed',
            user_id=1  # Will be associated after commit
        )
        
        db.session.add(sample_dataset)
        db.session.commit()
        
        print("✅ Database seeded with sample data!")
        print("   Demo user: demo@inferx.ml / demo123")


def reset_db():
    """Reset database (drop and recreate)"""
    print("⚠️ Resetting database...")
    drop_db()
    init_db()
    print("✅ Database reset complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Database management utility')
    parser.add_argument('command', choices=['init', 'drop', 'seed', 'reset'],
                        help='Command to execute')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        init_db()
    elif args.command == 'drop':
        drop_db()
    elif args.command == 'seed':
        seed_db()
    elif args.command == 'reset':
        reset_db()

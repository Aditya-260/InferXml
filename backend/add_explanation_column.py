"""
Add explanation_data column to experiments table
"""
from app import create_app, db
from sqlalchemy import text

app = create_app()

with app.app_context():
    try:
        # Add explanation_data column if not exists
        db.session.execute(text("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS explanation_data JSONB DEFAULT '{}'"))
        db.session.commit()
        print("✅ Column 'explanation_data' added successfully!")
    except Exception as e:
        print(f"Error: {e}")
        db.session.rollback()

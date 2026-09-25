"""
Add thinking_logs column to experiments table
"""
from app import create_app, db
from sqlalchemy import text

app = create_app()

with app.app_context():
    try:
        # Add thinking_logs column if not exists
        db.session.execute(text("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS thinking_logs TEXT DEFAULT ''"))
        db.session.commit()
        print("✅ Column 'thinking_logs' added successfully!")
    except Exception as e:
        print(f"Error: {e}")
        db.session.rollback()

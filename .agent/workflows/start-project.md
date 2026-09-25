---
description: How to start the InferX-ML project (frontend and backend)
---

# Starting InferX-ML Project

## Prerequisites
- Docker Desktop must be running

## Quick Start (Both Frontend & Backend)

1. Open terminal and navigate to the project folder:
```powershell
cd c:\Users\om\OneDrive\Attachments\Desktop\InferX_ML\InferX-ML
```

// turbo
2. Start all services (backend + frontend + databases):
```powershell
docker-compose up -d
```

3. Access the application:
   - **Frontend**: http://localhost:3000
   - **Backend API**: http://localhost:5000/api/health
   - **MinIO Console**: http://localhost:9001 (admin: minioadmin/minioadmin)

---

## Start Only Backend

// turbo
1. Start backend with dependencies:
```powershell
docker-compose up backend -d
```

This starts: PostgreSQL, Redis, MinIO, and Flask backend

---

## Start Only Frontend

// turbo
1. Start frontend (requires backend running):
```powershell
docker-compose up frontend -d
```

---

## Useful Commands

### View running containers:
```powershell
docker ps
```

### View backend logs:
```powershell
docker-compose logs -f backend
```

### View frontend logs:
```powershell
docker-compose logs -f frontend
```

### Stop all services:
```powershell
docker-compose down
```

### Stop and remove volumes (resets database):
```powershell
docker-compose down -v
```

### Restart a specific service:
```powershell
docker-compose restart backend
```

---

## First-Time Setup (Database Initialization)

If you get "Registration failed" or database errors on first run:

```powershell
docker exec inferx-backend python -c "from app import create_app, db; app = create_app(); app.app_context().push(); db.create_all(); print('Database tables created!')"
```

---

## Environment Variables (Optional)

Create a `.env` file in the InferX-ML folder for API keys:
```
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

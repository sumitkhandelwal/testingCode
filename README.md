```python
# ==============================================================================
# FILE: requirements.txt
# ==============================================================================
# This file lists all the Python packages required to run the application.
# Install them using: pip install -r requirements.txt
#
# Note: You also need to install the appropriate ODBC drivers for Sybase and
# SQL Server on the machine where this application will run.

fastapi
uvicorn[standard]
python-multipart
jinja2
pyodbc
pandas
openpyxl
passlib[bcrypt]
python-jose[cryptography]
# For the application's internal database
sqlalchemy
aqlite # Async driver for SQLite

# ==============================================================================
# FILE: config.ini
# ==============================================================================
# This is the configuration file.
# Admin can modify this to add/remove database connections and tables.
# IMPORTANT: For security, use environment variables or a secrets management
# system for credentials in a production environment.

[app_config]
SECRET_KEY = 'a_very_secret_key_that_should_be_changed'
ALGORITHM = HS256
ACCESS_TOKEN_EXPIRE_MINUTES = 60

[database_connections]
# Define your database connections here. The key (e.g., MSSQL_PROD) will
# appear in the dropdown.
# You must have the correct ODBC drivers installed.

# Example for MS SQL Server
MSSQL_PROD = {
    "driver": "{ODBC Driver 17 for SQL Server}",
    "server": "your_sql_server_name_or_ip",
    "port": "1433",
    "database": "your_prod_db",
    "uid": "your_username",
    "pwd": "your_password"
}

# Example for Sybase ASE (adjust driver name as needed)
SYBASE_DEV = {
    "driver": "{Adaptive Server Enterprise}",
    "server": "your_sybase_server_name",
    "port": "5000",
    "database": "your_dev_db",
    "uid": "your_username",
    "pwd": "your_password"
}


[database_tables]
# List the tables that should be accessible for each connection.
# The key must match a key from [database_connections].

MSSQL_PROD = [
    "Customers",
    "Orders",
    "Products"
]

SYBASE_DEV = [
    "sysusers",
    "sysobjects"
]


# ==============================================================================
# DIRECTORY: app/
# This directory will contain the main application logic.
# ==============================================================================

# ==============================================================================
# FILE: app/main.py
# ==============================================================================
# This is the entry point of the application.

import logging
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

from .core.config import settings
from .core.security import get_current_user_from_cookie
from .routers import auth, query_tool, compare
from .database.db_models import Base, engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables for users and audit logs
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Database Query Tool")

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(auth.router, tags=["Authentication"])
app.include_router(query_tool.router, tags=["Query Tool"])
app.include_router(compare.router, tags=["Comparison Tool"])


@app.get("/", response_class=HTMLResponse)
async def root(request: Request, user: dict = Depends(get_current_user_from_cookie)):
    """
    Root endpoint. Redirects to the dashboard if logged in,
    otherwise shows the login page.
    """
    if user:
        return RedirectResponse(url="/dashboard")
    return RedirectResponse(url="/login")

# ==============================================================================
# DIRECTORY: app/core/
# Core application logic like configuration and security.
# ==============================================================================

# ==============================================================================
# FILE: app/core/config.py
# ==============================================================================
import configparser
import json
from pathlib import Path

class Settings:
    def __init__(self, config_file: str = "config.ini"):
        config = configparser.ConfigParser()
        config.read(config_file)

        # App settings
        self.SECRET_KEY: str = config.get('app_config', 'SECRET_KEY')
        self.ALGORITHM: str = config.get('app_config', 'ALGORITHM')
        self.ACCESS_TOKEN_EXPIRE_MINUTES: int = config.getint('app_config', 'ACCESS_TOKEN_EXPIRE_MINUTES')

        # Database connections
        self.DB_CONNECTIONS = {}
        if 'database_connections' in config:
            for key, value in config.items('database_connections'):
                self.DB_CONNECTIONS[key.upper()] = json.loads(value.replace("'", "\""))

        # Accessible tables
        self.DB_TABLES = {}
        if 'database_tables' in config:
            for key, value in config.items('database_tables'):
                self.DB_TABLES[key.upper()] = json.loads(value.replace("'", "\""))

settings = Settings()

# ==============================================================================
# FILE: app/core/security.py
# ==============================================================================
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user_from_cookie(request: Request) -> Optional[dict]:
    """
    Dependency to get the current user from the access token in the cookie.
    Returns the user payload dict if token is valid, otherwise None.
    """
    token = request.cookies.get("access_token")
    if not token:
        return None

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # The token from cookie includes "bearer ", remove it
        token = token.split(" ")[1]
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return payload
    except JWTError:
        return None

# ==============================================================================
# DIRECTORY: app/database/
# For database models and connection logic.
# ==============================================================================

# ==============================================================================
# FILE: app/database/db_models.py
# ==============================================================================
# SQLAlchemy models for the application's internal database (users, audit log).

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///./app_internal.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="Business User") # Roles: Admin, IT User, Business User

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    action = Column(String)
    details = Column(Text, nullable=True)

# ==============================================================================
# FILE: app/database/db_actions.py
# ==============================================================================
# Functions to interact with the internal SQLite database.

from sqlalchemy.orm import Session
from . import db_models
from ..schemas import UserCreate
from ..core.security import get_password_hash

def get_user(db: Session, username: str):
    return db.query(db_models.User).filter(db_models.User.username == username).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = db_models.User(username=user.username, hashed_password=hashed_password, role=user.role)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def log_activity(db: Session, username: str, action: str, details: str = None):
    log_entry = db_models.AuditLog(username=username, action=action, details=details)
    db.add(log_entry)
    db.commit()

# ==============================================================================
# FILE: app/database/query_executor.py
# ==============================================================================
# Logic to connect to external databases (Sybase, MSSQL) and run queries.
import pyodbc
import pandas as pd
from ..core.config import settings
import logging

logger = logging.getLogger(__name__)

def get_db_connection(db_name: str):
    """Establishes a connection to the specified external database."""
    if db_name not in settings.DB_CONNECTIONS:
        raise ValueError(f"Database connection '{db_name}' not found in configuration.")
    
    conn_details = settings.DB_CONNECTIONS[db_name]
    
    conn_str = (
        f"DRIVER={conn_details['driver']};"
        f"SERVER={conn_details['server']};"
        f"DATABASE={conn_details['database']};"
        f"UID={conn_details['uid']};"
        f"PWD={conn_details['pwd']};"
    )
    
    try:
        conn = pyodbc.connect(conn_str)
        return conn
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        logger.error(f"DATABASE CONNECTION ERROR: {sqlstate} - {ex}")
        raise ConnectionError(f"Failed to connect to {db_name}. Please check configuration and network.")


def execute_query(db_name: str, query: str) -> pd.DataFrame:
    """Executes a SELECT query and returns results as a pandas DataFrame."""
    if not query.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    try:
        with get_db_connection(db_name) as conn:
            df = pd.read_sql(query, conn)
            # Sanitize column names for HTML display
            df.columns = [str(col).replace(' ', '_') for col in df.columns]
            return df
    except Exception as e:
        logger.error(f"QUERY EXECUTION FAILED for '{db_name}': {e}")
        raise


# ==============================================================================
# DIRECTORY: app/schemas/
# Pydantic models for data validation.
# ==============================================================================

# ==============================================================================
# FILE: app/schemas.py
# ==============================================================================
from pydantic import BaseModel
from typing import List, Optional

class UserBase(BaseModel):
    username: str
    role: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class QueryRequest(BaseModel):
    db_name: str
    table_name: str

class NlpQueryRequest(BaseModel):
    db_name: str
    table_name: str
    natural_language_query: str
    
class CompareRequest(BaseModel):
    db1_name: str
    table1_name: str
    db2_name: str
    table2_name: str
    join_key: str


# ==============================================================================
# DIRECTORY: app/routers/
# API endpoint definitions.
# ==============================================================================

# ==============================================================================
# FILE: app/routers/auth.py
# ==============================================================================
from fastapi import APIRouter, Depends, HTTPException, status, Request, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .. import schemas
from ..core.security import verify_password, create_access_token, get_current_user_from_cookie
from ..database import db_actions, db_models
from datetime import timedelta

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def get_db():
    db = db_models.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # For simplicity, we'll create a default admin user if one doesn't exist.
    # In a real app, this would be handled by a setup script.
    db = next(get_db())
    if not db_actions.get_user(db, "admin"):
        from ..core.security import get_password_hash
        admin_user = db_models.User(username="admin", hashed_password=get_password_hash("admin"), role="Admin")
        db.add(admin_user)
        db.commit()
    db.close()
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
async def login_for_access_token(response: Response, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db_actions.get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}
    )
    
    # Set token in a secure, HTTPOnly cookie
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=f"bearer {access_token}", httponly=True)
    
    db_actions.log_activity(db, username, "User Login")
    return response

@router.get("/logout")
async def logout(response: Response):
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response

# Placeholder for password reset page
@router.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(request: Request):
    return templates.TemplateResponse("reset_password.html", {"request": request})


# ==============================================================================
# FILE: app/routers/query_tool.py
# ==============================================================================
from fastapi import APIRouter, Depends, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import pandas as pd

from ..core.config import settings
from ..core.security import get_current_user_from_cookie
from ..database import query_executor, db_actions, db_models
from ..services.nlp_service import convert_nlp_to_sql

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def get_db():
    db = db_models.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: dict = Depends(get_current_user_from_cookie)):
    if not user:
        return RedirectResponse(url="/login")
    
    db_names = list(settings.DB_CONNECTIONS.keys())
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "db_names": db_names,
        "user": user
    })

@router.get("/get-tables/{db_name}")
async def get_tables(db_name: str, user: dict = Depends(get_current_user_from_cookie)):
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    
    tables = settings.DB_TABLES.get(db_name.upper(), [])
    return JSONResponse(content={"tables": tables})

@router.post("/execute-query")
async def execute_table_query(
    request: Request,
    db_name: str = Form(...),
    table_name: str = Form(...),
    user: dict = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login")
    
    query = f"SELECT * FROM {table_name}"
    try:
        df = query_executor.execute_query(db_name, query)
        db_actions.log_activity(db, user['sub'], "Execute Query", f"DB: {db_name}, Query: {query}")
        return templates.TemplateResponse("partials/results_table.html", {
            "request": request,
            "data": df.to_dict(orient='records'),
            "columns": df.columns.tolist()
        })
    except Exception as e:
        return HTMLResponse(content=f"<div class='p-4 text-red-700 bg-red-100 border border-red-400 rounded'>Error: {e}</div>")

@router.post("/upload-sql")
async def upload_sql_file(
    request: Request,
    db_name: str = Form(...),
    sql_file: UploadFile = File(...),
    user: dict = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user or user.get("role") not in ["Admin", "IT User"]:
        return HTMLResponse(content="<div class='p-4 text-red-700 bg-red-100 border border-red-400 rounded'>Error: Insufficient permissions.</div>", status_code=403)

    query = (await sql_file.read()).decode("utf-8")
    
    if not query.strip().upper().startswith("SELECT"):
        return HTMLResponse(content="<div class='p-4 text-red-700 bg-red-100 border border-red-400 rounded'>Error: Only SELECT queries are allowed.</div>")

    try:
        df = query_executor.execute_query(db_name, query)
        db_actions.log_activity(db, user['sub'], "Upload SQL", f"DB: {db_name}, File: {sql_file.filename}")
        return templates.TemplateResponse("partials/results_table.html", {
            "request": request,
            "data": df.to_dict(orient='records'),
            "columns": df.columns.tolist()
        })
    except Exception as e:
        return HTMLResponse(content=f"<div class='p-4 text-red-700 bg-red-100 border border-red-400 rounded'>Error: {e}</div>")

@router.post("/execute-nlp-query")
async def execute_nlp_query(
    request: Request,
    db_name: str = Form(...),
    table_name: str = Form(...),
    nlp_query: str = Form(...),
    user: dict = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login")

    try:
        # This is a simplified NLP to SQL conversion.
        # A real-world implementation would require a more sophisticated model.
        sql_query = convert_nlp_to_sql(nlp_query, table_name)
        
        df = query_executor.execute_query(db_name, sql_query)
        db_actions.log_activity(db, user['sub'], "Execute NLP Query", f"DB: {db_name}, NLP: '{nlp_query}', SQL: '{sql_query}'")
        
        return templates.TemplateResponse("partials/results_table.html", {
            "request": request,
            "data": df.to_dict(orient='records'),
            "columns": df.columns.tolist(),
            "generated_sql": sql_query
        })
    except Exception as e:
        return HTMLResponse(content=f"<div class='p-4 text-red-700 bg-red-100 border border-red-400 rounded'>Error: {e}</div>")

# ==============================================================================
# FILE: app/routers/compare.py
# ==============================================================================
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import pandas as pd

from ..core.config import settings
from ..core.security import get_current_user_from_cookie
from ..database import query_executor, db_actions, db_models
from ..services.compare_service import perform_comparison

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def get_db():
    db = db_models.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request, user: dict = Depends(get_current_user_from_cookie)):
    if not user:
        return RedirectResponse(url="/login")
    
    db_names = list(settings.DB_CONNECTIONS.keys())
    return templates.TemplateResponse("compare.html", {
        "request": request,
        "db_names": db_names,
        "user": user
    })

@router.post("/run-comparison")
async def run_comparison(
    request: Request,
    db1_name: str = Form(...),
    table1_name: str = Form(...),
    db2_name: str = Form(...),
    table2_name: str = Form(...),
    join_key: str = Form(...),
    user: dict = Depends(get_current_user_from_cookie),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login")
    
    try:
        comparison_result = perform_comparison(db1_name, table1_name, db2_name, table2_name, join_key)
        
        log_details = f"Compared {db1_name}.{table1_name} with {db2_name}.{table2_name} on key '{join_key}'"
        db_actions.log_activity(db, user['sub'], "Run Comparison", log_details)

        return templates.TemplateResponse("partials/compare_results.html", {
            "request": request,
            "result": comparison_result
        })
    except Exception as e:
        return HTMLResponse(content=f"<div class='p-4 text-red-700 bg-red-100 border border-red-400 rounded'>Error: {e}</div>")


# ==============================================================================
# DIRECTORY: app/services/
# Business logic services.
# ==============================================================================

# ==============================================================================
# FILE: app/services/nlp_service.py
# ==============================================================================
# A very basic NLP to SQL converter.
# This is a placeholder and should be replaced with a more robust solution.

def convert_nlp_to_sql(text: str, table_name: str) -> str:
    """
    Simplistic NLP to SQL conversion.
    Example: "show me all customers from New York"
    -> SELECT * FROM Customers WHERE city = 'New York'
    """
    text = text.lower()
    
    # Basic keyword mapping
    if "show me all" in text or "get all" in text or "list all" in text:
        query = f"SELECT * FROM {table_name}"
        
        # Simple "where" clause detection
        if "from" in text and "with" not in text:
             # e.g., "from New York" -> assumes a 'city' column
            parts = text.split("from")
            if len(parts) > 1:
                location = parts[1].strip().title()
                query += f" WHERE city = '{location}'" # Dangerous! Prone to SQLi. Use parameterized queries.
                
        if "with" in text:
            # e.g., "with more than 5 orders" -> assumes an 'order_count' column
            parts = text.split("with")
            if len(parts) > 1 and "more than" in parts[1]:
                num_part = parts[1].split("more than")[1].strip()
                try:
                    num = int(num_part.split(' ')[0])
                    query += f" WHERE order_count > {num}"
                except ValueError:
                    pass # Ignore if not a number
        return query

    # Default fallback
    return f"SELECT * FROM {table_name}"

# ==============================================================================
# FILE: app/services/compare_service.py
# ==============================================================================
import pandas as pd
from ..database import query_executor

def perform_comparison(db1_name, table1_name, db2_name, table2_name, join_key):
    """
    Performs a comparison between two tables from potentially different databases.
    """
    query1 = f"SELECT * FROM {table1_name}"
    query2 = f"SELECT * FROM {table2_name}"
    
    df1 = query_executor.execute_query(db1_name, query1)
    df2 = query_executor.execute_query(db2_name, query2)

    # Ensure join key exists in both dataframes
    if join_key not in df1.columns or join_key not in df2.columns:
        raise ValueError(f"Join key '{join_key}' not found in one or both tables.")

    # Perform the merge
    merged_df = pd.merge(df1, df2, on=join_key, how='outer', suffixes=('_1', '_2'), indicator=True)

    # Analysis
    count1 = len(df1)
    count2 = len(df2)
    
    only_in_1 = merged_df[merged_df['_merge'] == 'left_only']
    only_in_2 = merged_df[merged_df['_merge'] == 'right_only']
    in_both = merged_df[merged_df['_merge'] == 'both']

    # Find mismatches in common records
    mismatches = []
    common_cols = [col for col in df1.columns if col in df2.columns and col != join_key]
    
    for _, row in in_both.iterrows():
        mismatch_details = {}
        for col in common_cols:
            val1 = row[f'{col}_1']
            val2 = row[f'{col}_2']
            if pd.notna(val1) and pd.notna(val2) and val1 != val2:
                mismatch_details[col] = {'val1': val1, 'val2': val2}
        
        if mismatch_details:
            mismatches.append({
                "key": row[join_key],
                "mismatches": mismatch_details
            })

    return {
        "summary": {
            "source1_count": count1,
            "source2_count": count2,
            "common_records": len(in_both),
            "only_in_source1_count": len(only_in_1),
            "only_in_source2_count": len(only_in_2),
            "mismatched_records": len(mismatches)
        },
        "mismatches": mismatches,
        "only_in_1": only_in_1[[join_key] + common_cols].to_dict(orient='records'),
        "only_in_2": only_in_2[[join_key] + common_cols].to_dict(orient='records')
    }

# ==============================================================================
# DIRECTORY: static/
# For CSS and JavaScript files.
# ==============================================================================

# ==============================================================================
# FILE: static/style.css
# ==============================================================================
/* Using Tailwind CSS via CDN in the HTML, but you can add custom styles here */
body {
    font-family: 'Inter', sans-serif;
    background-color: #f7fafc; /* gray-100 */
}

.table-container {
    max-height: 60vh;
    overflow-y: auto;
}

table th {
    background-color: #f2f2f2; /* gray-200 */
    position: sticky;
    top: 0;
}

/* Simple loading spinner */
.loader {
    border: 4px solid #f3f3f3; /* Light grey */
    border-top: 4px solid #EF241C; /* Custom Red */
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 2rem auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}


# ==============================================================================
# DIRECTORY: templates/
# For HTML templates (Jinja2).
# ==============================================================================

# ==============================================================================
# FILE: templates/base.html
# ==============================================================================
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Database Query Tool{% endblock %}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
    <link rel="stylesheet" href="/static/style.css">
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <style>
      /* Custom styles to inject the new red color */
      .bg-primary { background-color: #EF241C; }
      .text-primary { color: #EF241C; }
      .btn-primary {
        background-color: #EF241C;
        color: white;
      }
      .btn-primary:hover {
        background-color: #d41e17; /* Darker shade */
      }
      .hover-bg-primary-dark:hover {
        background-color: #d41e17;
      }
      .text-primary-dark {
        color: #d41e17;
      }
      .bg-primary-lighter {
        background-color: #fee2e2;
      }
      .bg-primary-lightest {
        background-color: #fef2f2;
      }
      input[type="file"]::file-selector-button {
          margin-right: 1rem; padding: 0.5rem 1rem; border-radius: 9999px;
          border-width: 0; font-size: 0.875rem; font-weight: 600;
          background-color: #fef2f2; color: #d41e17; cursor: pointer;
          transition: background-color 0.2s;
      }
      input[type="file"]::file-selector-button:hover {
            background-color: #fee2e2;
      }
    </style>
</head>
<body class="bg-gray-100">

    <header class="bg-primary text-white shadow-md">
        <nav class="container mx-auto px-6 py-3 flex justify-between items-center">
            <div class="flex items-center">
                <!-- Logo Placeholder -->
                <div class="w-10 h-10 bg-white rounded-full mr-3"></div>
                <a class="text-xl font-semibold" href="/">Regulator Query Tool</a>
            </div>
            <div>
                {% if user %}
                    <a href="/dashboard" class="px-3 py-2 rounded hover-bg-primary-dark">Dashboard</a>
                    <a href="/compare" class="px-3 py-2 rounded hover-bg-primary-dark">Compare Data</a>
                    <span class="px-3 py-2">Welcome, {{ user.sub }} ({{ user.role }})</span>
                    <a href="/logout" class="px-3 py-2 rounded bg-white text-primary-dark font-bold hover:bg-gray-200">Logout</a>
                {% endif %}
            </div>
        </nav>
    </header>

    <main class="container mx-auto p-6">
        {% block content %}{% endblock %}
    </main>

    <footer class="bg-white text-center text-sm py-4 mt-8 border-t">
        &copy; 2025 Regulator Corp. All rights reserved.
    </footer>

    {% block scripts %}{% endblock %}
</body>
</html>

# ==============================================================================
# FILE: templates/login.html
# ==============================================================================
{% extends "base.html" %}

{% block title %}Login{% endblock %}

{% block content %}
<div class="flex items-center justify-center min-h-[60vh]">
    <div class="w-full max-w-md p-8 space-y-6 bg-white rounded-lg shadow-md">
        <h2 class="text-2xl font-bold text-center text-gray-800">Secure Login</h2>
        <form action="/login" method="post" class="space-y-6">
            <div>
                <label for="username" class="text-sm font-semibold text-gray-700 block">Username</label>
                <input type="text" id="username" name="username" required
                       class="w-full px-4 py-2 mt-2 border rounded-md focus:outline-none focus:ring-1 focus:ring-red-600">
            </div>
            <div>
                <label for="password" class="text-sm font-semibold text-gray-700 block">Password</label>
                <input type="password" id="password" name="password" required
                       class="w-full px-4 py-2 mt-2 border rounded-md focus:outline-none focus:ring-1 focus:ring-red-600">
            </div>
            <div>
                <button type="submit" class="w-full btn-primary py-2">
                    Login
                </button>
            </div>
        </form>
        <div class="text-center">
            <a href="/reset-password" class="text-sm text-primary hover:underline">Forgot Password?</a>
        </div>
    </div>
</div>
{% endblock %}


# ==============================================================================
# FILE: templates/dashboard.html
# ==============================================================================
{% extends "base.html" %}

{% block title %}Dashboard{% endblock %}

{% block content %}
<div class="space-y-8">
    <!-- Query Builder Section -->
    <div class="bg-white p-6 rounded-lg shadow-md">
        <h2 class="text-xl font-semibold mb-4 text-gray-800 border-b pb-2">Query Builder</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Direct Table Query -->
            <form id="query-form"
                  hx-post="/execute-query"
                  hx-target="#results-container"
                  hx-indicator="#loader"
                  class="space-y-4">
                <h3 class="font-semibold text-gray-700">1. Select Table to Query</h3>
                <div>
                    <label for="db-select" class="block text-sm font-medium text-gray-700">Database</label>
                    <select id="db-select" name="db_name" class="mt-1 block w-full p-2 border border-gray-300 rounded-md"
                            hx-get="/get-tables/" hx-target="#table-select" hx-trigger="change">
                        <option value="">Select a Database</option>
                        {% for db in db_names %}
                        <option value="{{ db }}">{{ db }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div>
                    <label for="table-select" class="block text-sm font-medium text-gray-700">Table</label>
                    <select id="table-select" name="table_name" class="mt-1 block w-full p-2 border border-gray-300 rounded-md">
                        <option>Select a database first</option>
                    </select>
                </div>
                <button type="submit" class="btn-primary">Execute Query</button>
            </form>

            <!-- SQL File Upload -->
            {% if user.role in ['Admin', 'IT User'] %}
            <form id="upload-form"
                  hx-post="/upload-sql"
                  hx-target="#results-container"
                  hx-encoding="multipart/form-data"
                  hx-indicator="#loader"
                  class="space-y-4">
                <h3 class="font-semibold text-gray-700">2. Or Upload a .sql File</h3>
                <input type="hidden" name="db_name" id="upload-db-name">
                <div>
                    <label for="sql-file" class="block text-sm font-medium text-gray-700">SQL File (SELECT only)</label>
                    <input type="file" name="sql_file" id="sql-file" required class="mt-1 block w-full text-sm text-gray-500">
                </div>
                <button type="submit" class="btn-primary">Upload and Execute</button>
            </form>
            {% endif %}
        </div>
    </div>

    <!-- Natural Language Query Section -->
    <div class="bg-white p-6 rounded-lg shadow-md">
        <h2 class="text-xl font-semibold mb-4 text-gray-800 border-b pb-2">Natural Language Query</h2>
        <form id="nlp-form"
              hx-post="/execute-nlp-query"
              hx-target="#results-container"
              hx-indicator="#loader"
              class="space-y-4">
            <input type="hidden" name="db_name" id="nlp-db-name">
            <input type="hidden" name="table_name" id="nlp-table-name">
            <div>
                <label for="nlp-query" class="block text-sm font-medium text-gray-700">Enter your query in plain English</label>
                <textarea id="nlp-query" name="nlp_query" rows="3" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md" placeholder="e.g., show me all customers from New York with more than 5 orders"></textarea>
            </div>
            <button type="submit" class="btn-primary">Translate and Execute</button>
        </form>
    </div>

    <!-- Results Section -->
    <div class="bg-white p-6 rounded-lg shadow-md min-h-[200px]">
        <h2 class="text-xl font-semibold mb-4 text-gray-800">Results</h2>
        <div id="loader" class="htmx-indicator loader"></div>
        <div id="results-container">
            <p class="text-gray-500">Query results will be displayed here.</p>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    // Sync the database selection across forms
    document.getElementById('db-select').addEventListener('change', function() {
        const selectedDb = this.value;
        document.getElementById('upload-db-name').value = selectedDb;
        document.getElementById('nlp-db-name').value = selectedDb;
    });
    // Sync table selection for NLP form
    document.getElementById('table-select').addEventListener('change', function() {
        document.getElementById('nlp-table-name').value = this.value;
    });
</script>
{% endblock %}


# ==============================================================================
# FILE: templates/compare.html
# ==============================================================================
{% extends "base.html" %}

{% block title %}Data Comparison{% endblock %}

{% block content %}
<div class="space-y-8">
    <!-- Comparison Setup Section -->
    <div class="bg-white p-6 rounded-lg shadow-md">
        <h2 class="text-xl font-semibold mb-4 text-gray-800 border-b pb-2">Data Comparison Tool</h2>
        <form id="compare-form"
              hx-post="/run-comparison"
              hx-target="#compare-results-container"
              hx-indicator="#compare-loader"
              class="space-y-6">

            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <!-- Source 1 -->
                <div class="space-y-4 border-r pr-8">
                    <h3 class="font-semibold text-lg text-gray-700">Source 1</h3>
                    <div>
                        <label for="db1-select" class="block text-sm font-medium text-gray-700">Database</label>
                        <select id="db1-select" name="db1_name" class="mt-1 block w-full p-2 border border-gray-300 rounded-md"
                                hx-get="/get-tables/" hx-target="#table1-select" hx-trigger="change">
                            <option value="">Select Database</option>
                            {% for db in db_names %}
                            <option value="{{ db }}">{{ db }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div>
                        <label for="table1-select" class="block text-sm font-medium text-gray-700">Table</label>
                        <select id="table1-select" name="table1_name" class="mt-1 block w-full p-2 border border-gray-300 rounded-md">
                             <option>Select a database first</option>
                        </select>
                    </div>
                </div>

                <!-- Source 2 -->
                <div class="space-y-4">
                    <h3 class="font-semibold text-lg text-gray-700">Source 2</h3>
                    <div>
                        <label for="db2-select" class="block text-sm font-medium text-gray-700">Database</label>
                        <select id="db2-select" name="db2_name" class="mt-1 block w-full p-2 border border-gray-300 rounded-md"
                                hx-get="/get-tables/" hx-target="#table2-select" hx-trigger="change">
                            <option value="">Select Database</option>
                            {% for db in db_names %}
                            <option value="{{ db }}">{{ db }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div>
                        <label for="table2-select" class="block text-sm font-medium text-gray-700">Table</label>
                        <select id="table2-select" name="table2_name" class="mt-1 block w-full p-2 border border-gray-300 rounded-md">
                            <option>Select a database first</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Join Key and Submit -->
            <div class="pt-6 border-t">
                <label for="join-key" class="block text-sm font-medium text-gray-700">Join Key / Common Column Name</label>
                <input type="text" id="join-key" name="join_key" required placeholder="e.g., customer_id"
                       class="mt-1 block w-full md:w-1/3 px-4 py-2 border rounded-md focus:outline-none focus:ring-1 focus:ring-red-600">
            </div>
            
            <button type="submit" class="btn-primary">Run Comparison</button>
        </form>
    </div>

    <!-- Comparison Results Section -->
    <div class="bg-white p-6 rounded-lg shadow-md min-h-[200px]">
        <h2 class="text-xl font-semibold mb-4 text-gray-800">Comparison Results</h2>
        <div id="compare-loader" class="htmx-indicator loader"></div>
        <div id="compare-results-container">
            <p class="text-gray-500">Comparison analysis will be displayed here.</p>
        </div>
    </div>
</div>
{% endblock %}

# ==============================================================================
# FILE: templates/reset_password.html
# ==============================================================================
{% extends "base.html" %}

{% block title %}Reset Password{% endblock %}

{% block content %}
<div class="flex items-center justify-center min-h-[60vh]">
    <div class="w-full max-w-md p-8 space-y-6 bg-white rounded-lg shadow-md">
        <h2 class="text-2xl font-bold text-center text-gray-800">Reset Password</h2>
        <p class="text-center text-gray-600">This feature is under construction. In a real application, you would enter your email here to receive a reset link.</p>
        <form class="space-y-6">
            <div>
                <label for="email" class="text-sm font-semibold text-gray-700 block">Email Address</label>
                <input type="email" id="email" name="email" disabled
                       class="w-full px-4 py-2 mt-2 border rounded-md bg-gray-100">
            </div>
            <div>
                <button type="submit" disabled class="w-full btn-primary opacity-50 cursor-not-allowed py-2">
                    Send Reset Link
                </button>
            </div>
        </form>
         <div class="text-center">
            <a href="/login" class="text-sm text-primary hover:underline">Back to Login</a>
        </div>
    </div>
</div>
{% endblock %}

# ==============================================================================
# FILE: templates/partials/results_table.html
# ==============================================================================
{% if generated_sql %}
<div class="mb-4 p-3 bg-blue-50 border border-blue-200 text-blue-800 rounded-md">
    <p class="font-semibold">Generated SQL Query:</p>
    <code class="text-sm">{{ generated_sql }}</code>
</div>
{% endif %}

{% if data %}
<div class="table-container border rounded-md">
    <table class="min-w-full divide-y divide-gray-200">
        <thead class="bg-gray-50">
            <tr>
                {% for col in columns %}
                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {{ col }}
                </th>
                {% endfor %}
            </tr>
        </thead>
        <tbody class="bg-white divide-y divide-gray-200">
            {% for row in data %}
            <tr>
                {% for col in columns %}
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{{ row[col] }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
<div class="mt-4">
    <p class="text-sm text-gray-600">Showing {{ data|length }} rows.</p>
    <!-- Add export button here in a real app -->
</div>
{% else %}
<p class="text-gray-500">The query returned no results.</p>
{% endif %}

# ==============================================================================
# FILE: templates/partials/compare_results.html
# ==============================================================================
{% if result %}
<div class="space-y-6">
    <!-- Summary Section -->
    <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 text-center">
        <div class="p-4 bg-blue-100 rounded-lg">
            <p class="text-2xl font-bold text-blue-800">{{ result.summary.source1_count }}</p>
            <p class="text-sm text-blue-700">Records in Source 1</p>
        </div>
        <div class="p-4 bg-blue-100 rounded-lg">
            <p class="text-2xl font-bold text-blue-800">{{ result.summary.source2_count }}</p>
            <p class="text-sm text-blue-700">Records in Source 2</p>
        </div>
        <div class="p-4 bg-green-100 rounded-lg">
            <p class="text-2xl font-bold text-green-800">{{ result.summary.common_records }}</p>
            <p class="text-sm text-green-700">Common Records</p>
        </div>
        <div class="p-4 bg-yellow-100 rounded-lg">
            <p class="text-2xl font-bold text-yellow-800">{{ result.summary.mismatched_records }}</p>
            <p class="text-sm text-yellow-700">Mismatched Records</p>
        </div>
         <div class="p-4 bg-primary-lighter rounded-lg">
            <p class="text-2xl font-bold text-primary-dark">{{ result.summary.only_in_source1_count + result.summary.only_in_source2_count }}</p>
            <p class="text-sm text-primary-dark">Unique Records</p>
        </div>
    </div>

    <!-- Detailed Mismatches -->
    {% if result.mismatches %}
    <div>
        <h3 class="text-lg font-semibold text-gray-800 mb-2">Mismatched Records ({{ result.summary.mismatched_records }})</h3>
        <div class="table-container border rounded-md">
            <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                    <tr>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Key</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Mismatched Column</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Source 1 Value</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Source 2 Value</th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                    {% for item in result.mismatches %}
                        {% for col, vals in item.mismatches.items() %}
                        <tr>
                            <td class="px-6 py-4 text-sm">{{ item.key }}</td>
                            <td class="px-6 py-4 text-sm font-semibold">{{ col }}</td>
                            <td class="px-6 py-4 text-sm bg-primary-lightest">{{ vals.val1 }}</td>
                            <td class="px-6 py-4 text-sm bg-green-50">{{ vals.val2 }}</td>
                        </tr>
                        {% endfor %}
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    {% endif %}
</div>
{% else %}
<p class="text-gray-500">An error occurred or no results to display.</p>
{% endif %}

```

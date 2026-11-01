from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=True)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # "pdf" or "docx"
    language = Column(String, nullable=True)     # "en" or "fr"
    indexing_status = Column(String, default="pending")  # pending, indexed, failed
    uploaded_at = Column(DateTime, default=datetime.utcnow)


# Creates the table in Postgres if it doesn't exist yet
Base.metadata.create_all(engine)
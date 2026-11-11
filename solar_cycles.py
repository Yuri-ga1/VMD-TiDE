from sqlalchemy import create_engine, Column, Integer, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Define the base class for ORM models
Base = declarative_base()

# Define the SolarCycles table
class SolarCycles(Base):
    __tablename__ = 'solar_cycles'
    __table_args__ = {'schema': os.getenv("DB_SCHEMA", "public")}  # Use schema from .env or default to 'public'

    id = Column(Integer, primary_key=True)       # Primary key column
    name = Column(Integer)                       # Name of the solar cycle (integer)
    starts = Column(Date, nullable=False)        # Start date of the solar cycle
    peak = Column(Date, nullable=False)          # Peak date of the solar cycle
    ends = Column(Date, nullable=False)          # End date of the solar cycle

# Function to set up the database and create tables
def setup_database(schema=None):
    """
    Set up the database connection and create tables.
    Reads credentials from environment variables.
    :param schema: Optional PostgreSQL schema name (default is 'public').
    """
    # Retrieve PostgreSQL credentials from environment variables
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")  # Default PostgreSQL port
    DB_NAME = os.getenv("DB_NAME")

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        raise ValueError("Missing required database credentials in .env file.")

    # Construct the database URL
    DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Create the database engine
    engine = create_engine(DB_URL, echo=True)  # Set echo=False in production

    # Bind the schema to the metadata if provided
    Base.metadata.schema = schema

    # Create all tables if they don't exist
    Base.metadata.create_all(engine)
    print("Tables created successfully!")
    return engine

# Example usage
if __name__ == "__main__":
    try:
        # Set up the database and create tables
        engine = setup_database(schema=os.getenv("DB_SCHEMA", "public"))  # Use schema from .env or default to 'public'

        # Verify that the table was created
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema=os.getenv("DB_SCHEMA", "public"))
        print("Tables in database:", tables)

    except Exception as e:
        print(f"An error occurred: {e}")

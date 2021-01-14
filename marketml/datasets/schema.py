from sqlalchemy import Column, Integer, String, BLOB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ConferenceCallChunk(Base):
    __tablename__ = 'conference_call_chunks'
    call_reference = Column(String(10), nullable=False)
    index = Column(Integer(), nullable=False)
    total = Column(Integer(), nullable=False)
    filepath = Column(String(), nullable=False, unique=True, primary_key=True)
    url = Column(String(), nullable=False, unique=True)
    data = Column(BLOB(), nullable=True, unique=True)



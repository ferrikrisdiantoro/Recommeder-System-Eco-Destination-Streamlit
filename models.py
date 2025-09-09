from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    email = Column(String(180), nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)  # store as hex/base64 string
    created_at = Column(DateTime, default=datetime.utcnow)

class Place(Base):
    __tablename__ = "places"
    id = Column(Integer, primary_key=True)
    place_name = Column(String(255), nullable=False)
    place_description = Column(Text, default="")
    category = Column(String(255), default="")
    city = Column(String(120), default="")
    address = Column(String(255), default="")
    price_num = Column(Float, default=0.0)
    price_str = Column(String(64), default="")
    rating_avg = Column(Float, default=0.0)
    image = Column(String(500), default="")
    gallery1 = Column(String(500), default="")
    gallery2 = Column(String(500), default="")
    gallery3 = Column(String(500), default="")
    map_url = Column(String(500), default="")

class Rating(Base):
    __tablename__ = "ratings"
    __table_args__ = (UniqueConstraint("user_id", "place_id", name="uq_rating_user_place"),)
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    place_id = Column(Integer, ForeignKey("places.id"), nullable=False, index=True)
    rating = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    place_id = Column(Integer, ForeignKey("places.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Bookmark(Base):
    __tablename__ = "bookmarks"
    __table_args__ = (UniqueConstraint("user_id", "place_id", name="uq_bookmark_user_place"),)
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    place_id = Column(Integer, ForeignKey("places.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

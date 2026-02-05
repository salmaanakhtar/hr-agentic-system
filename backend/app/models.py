from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime
from typing import Dict, Any

class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role_id = Column(Integer, ForeignKey("roles.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    role = relationship("Role")

class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, unique=True, index=True)
    department = Column(String, nullable=True)

    user = relationship("User")

class Manager(Base):
    __tablename__ = "managers"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, unique=True, index=True)
    department = Column(String, nullable=True)

    user = relationship("User")

class HR(Base):
    __tablename__ = "hrs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, unique=True, index=True)

    user = relationship("User")

class ApprovalState(Base):
    __tablename__ = "approval_states"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)


class WorkflowState(Base):
    __tablename__ = "workflow_states"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(String, unique=True, index=True)
    workflow_type = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    status = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, index=True)
    completed_at = Column(DateTime, nullable=True)

    # JSON fields for complex data
    data = Column(JSON, default=dict)
    workflow_metadata = Column(JSON, default=dict)
    steps = Column(JSON, default=list)
    current_step = Column(String, nullable=True)
    next_steps = Column(JSON, default=list)

    # Versioning and validation
    version = Column(Integer, default=1)
    checksum = Column(String, nullable=True)
    is_valid = Column(Boolean, default=True)

    # Lifecycle management
    ttl_hours = Column(Integer, default=24)
    cleanup_scheduled = Column(Boolean, default=False)

    # Relationships
    user = relationship("User")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "user_id": self.user_id,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "data": self.data or {},
            "metadata": self.workflow_metadata or {},
            "steps": self.steps or [],
            "current_step": self.current_step,
            "next_steps": self.next_steps or [],
            "version": self.version,
            "checksum": self.checksum,
            "is_valid": self.is_valid,
            "ttl_hours": self.ttl_hours,
            "cleanup_scheduled": self.cleanup_scheduled
        }
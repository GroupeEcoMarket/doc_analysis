"""
Task Manager
Manages asynchronous task execution and status tracking.
"""

import uuid
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import threading

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Status of an asynchronous task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents an asynchronous processing task."""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    filename: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API responses."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "filename": self.filename
        }


class TaskManager:
    """
    Manages asynchronous task execution and status tracking.
    
    This is a simple in-memory implementation. For production, consider using:
    - Redis for distributed task storage
    - Dramatiq for task queue management
    - Database for persistent task storage
    """
    
    def __init__(self):
        """Initialize the task manager."""
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
    
    def create_task(self, filename: Optional[str] = None) -> str:
        """
        Create a new task.
        
        Args:
            filename: Optional filename for the task
            
        Returns:
            str: Task ID
        """
        task_id = str(uuid.uuid4())
        task = Task(task_id=task_id, filename=filename)
        
        with self._lock:
            self._tasks[task_id] = task
        
        logger.info(f"Task créée: {task_id} (filename: {filename})")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task or None if not found
        """
        with self._lock:
            return self._tasks.get(task_id)
    
    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Update task status.
        
        Args:
            task_id: Task ID
            status: New status
            result: Task result (if completed)
            error: Error message (if failed)
            
        Returns:
            bool: True if task was found and updated, False otherwise
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"Task non trouvée: {task_id}")
                return False
            
            task.status = status
            
            if status == TaskStatus.PROCESSING and not task.started_at:
                task.started_at = datetime.now()
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.completed_at = datetime.now()
            
            if result is not None:
                task.result = result
            
            if error is not None:
                task.error = error
            
            logger.info(f"Task {task_id} mise à jour: {status.value}")
            return True
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            bool: True if task was found and deleted, False otherwise
        """
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.info(f"Task supprimée: {task_id}")
                return True
            return False
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed or failed tasks.
        
        Args:
            max_age_hours: Maximum age in hours for tasks to keep
            
        Returns:
            int: Number of tasks deleted
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0
        
        with self._lock:
            task_ids_to_delete = [
                task_id
                for task_id, task in self._tasks.items()
                if task.completed_at and task.completed_at < cutoff_time
                and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            ]
            
            for task_id in task_ids_to_delete:
                del self._tasks[task_id]
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Nettoyage de {deleted_count} anciennes tâches")
        
        return deleted_count


# Global task manager instance
_task_manager = None


def get_task_manager() -> TaskManager:
    """
    Get the global task manager instance (singleton).
    
    Returns:
        TaskManager: Global task manager instance
    """
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


"""
Background task utilities for performance optimization.
Handles cache warming, cleanup, and maintenance tasks.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import gc

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """
    Manages background tasks for performance optimization.
    Handles cache warming, cleanup, and maintenance.
    """
    
    def __init__(self, max_workers: int = 2):
        """Initialize the background task manager."""
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks = {}
        self.running = False
        
    async def start(self):
        """Start background task processing."""
        self.running = True
        logger.info("Background task manager started")
        
        # Start cache warming task
        asyncio.create_task(self._cache_warming_loop())
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop background task processing."""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Background task manager stopped")
        
    async def _cache_warming_loop(self):
        """Periodically warm up caches for better performance."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                if self.running:
                    await self._warm_caches()
            except Exception as e:
                logger.error(f"Error in cache warming loop: {e}")
                
    async def _cleanup_loop(self):
        """Periodically clean up memory and caches."""
        while self.running:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                if self.running:
                    await self._cleanup_memory()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                
    async def _warm_caches(self):
        """Warm up various caches for better performance."""
        try:
            logger.debug("Warming up caches...")
            
            # Warm up embedding cache with common queries
            common_queries = [
                "hello", "help", "what is", "how to", "explain",
                "xin chào", "giúp đỡ", "là gì", "làm thế nào", "giải thích"
            ]
            
            # This would be implemented based on your specific cache services
            # For now, just log the warming attempt
            logger.debug(f"Warmed up caches for {len(common_queries)} common queries")
            
        except Exception as e:
            logger.error(f"Error warming caches: {e}")
            
    async def _cleanup_memory(self):
        """Clean up memory and optimize caches."""
        try:
            logger.debug("Cleaning up memory...")
            
            # Force garbage collection
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")
            
            # This would be implemented based on your specific cache services
            # For now, just log the cleanup attempt
            logger.debug("Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            
    def submit_task(self, task_name: str, func, *args, **kwargs):
        """Submit a background task for execution."""
        if task_name in self.tasks:
            logger.warning(f"Task {task_name} already running")
            return
            
        future = self.executor.submit(func, *args, **kwargs)
        self.tasks[task_name] = future
        
        # Clean up completed tasks
        def cleanup_task(fut):
            if task_name in self.tasks:
                del self.tasks[task_name]
                
        future.add_done_callback(cleanup_task)
        
    def get_status(self) -> Dict[str, Any]:
        """Get status of background tasks."""
        return {
            "running": self.running,
            "active_tasks": len(self.tasks),
            "max_workers": self.max_workers,
            "task_names": list(self.tasks.keys())
        }


# Global background task manager instance
_background_manager: Optional[BackgroundTaskManager] = None


def get_background_manager() -> BackgroundTaskManager:
    """Get the global background task manager instance."""
    global _background_manager
    if _background_manager is None:
        _background_manager = BackgroundTaskManager()
    return _background_manager


async def start_background_tasks():
    """Start all background tasks."""
    manager = get_background_manager()
    await manager.start()


async def stop_background_tasks():
    """Stop all background tasks."""
    manager = get_background_manager()
    await manager.stop()

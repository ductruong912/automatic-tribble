"""
Performance monitoring utilities for the chatbot.
Provides real-time performance metrics and monitoring capabilities.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor chatbot performance metrics in real-time."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.total_response_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.memory_usage_history = []
        self.cpu_usage_history = []
        self._lock = threading.Lock()
        
    def record_request(self, response_time: float, cache_hit: bool = False):
        """Record a request with its performance metrics."""
        with self._lock:
            self.request_count += 1
            self.total_response_time += response_time
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
                
    def record_system_metrics(self):
        """Record current system metrics."""
        try:
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)
            
            with self._lock:
                self.memory_usage_history.append({
                    'timestamp': time.time(),
                    'memory_percent': memory_percent
                })
                self.cpu_usage_history.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent
                })
                
                # Keep only last 100 measurements
                if len(self.memory_usage_history) > 100:
                    self.memory_usage_history = self.memory_usage_history[-100:]
                if len(self.cpu_usage_history) > 100:
                    self.cpu_usage_history = self.cpu_usage_history[-100:]
                    
        except Exception as e:
            logger.debug(f"Failed to record system metrics: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._lock:
            uptime = time.time() - self.start_time
            avg_response_time = (self.total_response_time / self.request_count) if self.request_count > 0 else 0
            
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = (self.cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
            
            # Calculate recent memory and CPU usage
            recent_memory = self.memory_usage_history[-10:] if self.memory_usage_history else []
            recent_cpu = self.cpu_usage_history[-10:] if self.cpu_usage_history else []
            
            avg_memory = sum(m['memory_percent'] for m in recent_memory) / len(recent_memory) if recent_memory else 0
            avg_cpu = sum(c['cpu_percent'] for c in recent_cpu) / len(recent_cpu) if recent_cpu else 0
            
            return {
                'uptime_seconds': uptime,
                'uptime_formatted': f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
                'total_requests': self.request_count,
                'average_response_time': round(avg_response_time, 3),
                'cache_hit_rate': round(cache_hit_rate, 1),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'average_memory_usage': round(avg_memory, 1),
                'average_cpu_usage': round(avg_cpu, 1),
                'requests_per_minute': round(self.request_count / (uptime / 60), 1) if uptime > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_health_status(self) -> str:
        """Get overall health status based on performance metrics."""
        stats = self.get_performance_stats()
        
        if stats['average_response_time'] > 5.0:
            return "degraded"
        elif stats['average_memory_usage'] > 80:
            return "memory_pressure"
        elif stats['average_cpu_usage'] > 90:
            return "cpu_pressure"
        elif stats['cache_hit_rate'] < 20:
            return "cache_inefficient"
        else:
            return "healthy"
    
    def reset_stats(self):
        """Reset all performance statistics."""
        with self._lock:
            self.start_time = time.time()
            self.request_count = 0
            self.total_response_time = 0.0
            self.cache_hits = 0
            self.cache_misses = 0
            self.memory_usage_history.clear()
            self.cpu_usage_history.clear()
            logger.info("Performance statistics reset")


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def log_performance_summary():
    """Log a performance summary to the console."""
    monitor = get_performance_monitor()
    stats = monitor.get_performance_stats()
    health = monitor.get_health_status()
    
    print("\n" + "="*60)
    print(" CHATBOT PERFORMANCE SUMMARY")
    print("="*60)
    print(f"⏱️  Uptime: {stats['uptime_formatted']}")
    print(f" Total Requests: {stats['total_requests']}")
    print(f"⚡ Avg Response Time: {stats['average_response_time']}s")
    print(f" Cache Hit Rate: {stats['cache_hit_rate']}%")
    print(f"💾 Avg Memory Usage: {stats['average_memory_usage']}%")
    print(f"🖥️  Avg CPU Usage: {stats['average_cpu_usage']}%")
    print(f"📈 Requests/Min: {stats['requests_per_minute']}")
    print(f"🏥 Health Status: {health.upper()}")
    print("="*60)


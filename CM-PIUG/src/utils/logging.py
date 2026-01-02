"""
Logging Utilities
=================

Provides consistent logging configuration for CM-PIUG framework.
"""

import logging
import sys
from typing import Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to output to console
        format_str: Custom format string
        
    Returns:
        Configured root logger
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Get root logger
    root_logger = logging.getLogger("cm_piug")
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name (usually module name)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"cm_piug.{name}")


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Convenience function to set up and get a named logger.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    setup_logging(level=level, console=True)
    return get_logger(name)


class ProgressLogger:
    """
    Logger for tracking algorithm progress.
    
    Attributes:
        name: Logger name
        logger: Underlying logger
        start_time: Start timestamp
    """
    
    def __init__(self, name: str, total_steps: Optional[int] = None):
        """
        Initialize progress logger.
        
        Args:
            name: Logger name
            total_steps: Total number of steps (optional)
        """
        self.name = name
        self.logger = get_logger(name)
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
    
    def start(self, message: str = "Starting..."):
        """Start progress tracking."""
        self.start_time = datetime.now()
        self.current_step = 0
        self.logger.info(f"[{self.name}] {message}")
    
    def step(self, message: str = "", increment: int = 1):
        """
        Record a step.
        
        Args:
            message: Step message
            increment: Number of steps to increment
        """
        self.current_step += increment
        if self.total_steps:
            pct = 100 * self.current_step / self.total_steps
            self.logger.debug(f"[{self.name}] Step {self.current_step}/{self.total_steps} ({pct:.1f}%) {message}")
        else:
            self.logger.debug(f"[{self.name}] Step {self.current_step} {message}")
    
    def finish(self, message: str = "Complete"):
        """Finish progress tracking."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        self.logger.info(f"[{self.name}] {message} (elapsed: {elapsed:.2f}s)")
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log an error."""
        if exception:
            self.logger.error(f"[{self.name}] {message}: {exception}")
        else:
            self.logger.error(f"[{self.name}] {message}")


class MetricsLogger:
    """
    Logger for tracking metrics during evaluation.
    """
    
    def __init__(self, name: str):
        """Initialize metrics logger."""
        self.name = name
        self.logger = get_logger(f"{name}.metrics")
        self.metrics = {}
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append((step, value))
        
        if step is not None:
            self.logger.info(f"[Step {step}] {key}: {value:.4f}")
        else:
            self.logger.info(f"{key}: {value:.4f}")
    
    def log_metrics(self, metrics_dict: dict, step: Optional[int] = None):
        """Log multiple metrics."""
        for key, value in metrics_dict.items():
            self.log_metric(key, value, step)
    
    def get_history(self, key: str) -> list:
        """Get metric history."""
        return self.metrics.get(key, [])
    
    def summary(self):
        """Print metrics summary."""
        self.logger.info("=" * 50)
        self.logger.info("Metrics Summary:")
        for key, values in self.metrics.items():
            if values:
                vals = [v for _, v in values]
                self.logger.info(f"  {key}: min={min(vals):.4f}, max={max(vals):.4f}, "
                               f"mean={sum(vals)/len(vals):.4f}, last={vals[-1]:.4f}")
        self.logger.info("=" * 50)

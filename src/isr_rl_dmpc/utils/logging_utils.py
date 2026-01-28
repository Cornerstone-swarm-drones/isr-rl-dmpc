"""
Logging utilities for ISR-RL-DMPC.

Provides structured logging infrastructure for mission monitoring,
debugging, and performance tracking.
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        """Format log record with colors."""
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        """Format log record as JSON."""
        log_dict = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_dict['exception'] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_dict.update(record.extra)

        return json.dumps(log_dict)


class MetricsLogger:
    """Logger for numerical metrics and statistics."""

    def __init__(self, name: str = 'metrics', log_dir: str = './logs'):
        """
        Initialize metrics logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # CSV file for metrics
        self.csv_path = self.log_dir / f"{name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_handler = None
        self.csv_file = None
        self.csv_headers = None
        self.csv_written = False

    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log dictionary of metrics to CSV.

        Args:
            metrics: Dictionary of metric_name -> value
            step: Optional step/episode number
        """
        if step is not None:
            metrics = {'step': step, **metrics}

        # Initialize CSV on first call
        if not self.csv_written:
            self.csv_headers = list(metrics.keys())
            with open(self.csv_path, 'w') as f:
                f.write(','.join(self.csv_headers) + '\n')
            self.csv_written = True

        # Write row
        with open(self.csv_path, 'a') as f:
            values = [str(metrics.get(h, '')) for h in self.csv_headers]
            f.write(','.join(values) + '\n')

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """
        Log scalar value.

        Args:
            name: Metric name
            value: Scalar value
            step: Step/episode number
        """
        self.log_dict({name: value}, step=step)

    def log_array(self, name: str, array: Any, step: int) -> None:
        """
        Log array value (as JSON).

        Args:
            name: Metric name
            array: Array to log
            step: Step/episode number
        """
        if hasattr(array, 'tolist'):
            array = array.tolist()
        self.log_dict({name: json.dumps(array)}, step=step)


class MissionLogger:
    """Logger for mission events and state changes."""

    def __init__(self, name: str = 'mission', log_dir: str = './logs'):
        """
        Initialize mission logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # File handler (detailed)
        file_path = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler (info and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def mission_started(self, mission_id: str, n_drones: int, area_size: tuple) -> None:
        """Log mission start event."""
        self.logger.info(
            f"Mission {mission_id} started with {n_drones} drones, "
            f"area size: {area_size}"
        )

    def mission_completed(self, mission_id: str, duration: float, coverage: float,
                         energy_used: float) -> None:
        """Log mission completion."""
        self.logger.info(
            f"Mission {mission_id} completed in {duration:.2f}s, "
            f"coverage: {coverage:.1f}%, energy used: {energy_used:.1f} Wh"
        )

    def drone_deployed(self, drone_id: int, position: tuple) -> None:
        """Log drone deployment."""
        self.logger.debug(f"Drone {drone_id} deployed at {position}")

    def drone_failed(self, drone_id: int, reason: str) -> None:
        """Log drone failure."""
        self.logger.error(f"Drone {drone_id} failed: {reason}")

    def target_detected(self, target_id: str, position: tuple,
                       classification: str, confidence: float) -> None:
        """Log target detection."""
        self.logger.info(
            f"Target {target_id} detected at {position}, "
            f"classification: {classification} (confidence: {confidence:.2f})"
        )

    def target_engaged(self, target_id: str, drone_id: int,
                      engagement_type: str) -> None:
        """Log target engagement."""
        self.logger.warning(
            f"Drone {drone_id} engaged target {target_id} "
            f"(type: {engagement_type})"
        )

    def coverage_milestone(self, coverage_pct: float, time_elapsed: float) -> None:
        """Log coverage milestone."""
        self.logger.info(f"Coverage milestone: {coverage_pct:.1f}% at {time_elapsed:.1f}s")

    def formation_event(self, event_type: str, n_drones: int, info: str = "") -> None:
        """Log formation-related event."""
        self.logger.info(f"Formation event: {event_type} ({n_drones} drones) - {info}")

    def error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log error."""
        if exception:
            self.logger.error(message, exc_info=exception)
        else:
            self.logger.error(message)

    def warning(self, message: str) -> None:
        """Log warning."""
        self.logger.warning(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message:str) -> None:
        """Log info message."""
        self.logger.info(message)


class PerformanceLogger:
    """Logger for performance profiling and benchmarking."""

    def __init__(self, name: str = 'performance', log_dir: str = './logs'):
        """
        Initialize performance logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.timings = {}  # Dict[timer_name, list of durations]
        self.csv_path = self.log_dir / f"{name}_timings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    def start_timer(self, timer_name: str) -> 'Timer':
        """
        Create a context manager timer.

        Usage:
            with perf_logger.start_timer('module_1'):
                # code to time
                pass

        Args:
            timer_name: Name of the timer

        Returns:
            Timer context manager
        """
        return Timer(self, timer_name)

    def record_timing(self, timer_name: str, duration: float) -> None:
        """
        Record a timing measurement.

        Args:
            timer_name: Name of the timer
            duration: Duration in seconds
        """
        if timer_name not in self.timings:
            self.timings[timer_name] = []
        self.timings[timer_name].append(duration)

    def get_statistics(self, timer_name: str) -> Dict[str, float]:
        """
        Get statistics for a timer.

        Args:
            timer_name: Name of the timer

        Returns:
            Dict with 'count', 'mean', 'min', 'max', 'total'
        """
        if timer_name not in self.timings:
            return {}

        timings = self.timings[timer_name]
        import numpy as np

        return {
            'count': len(timings),
            'mean': np.mean(timings),
            'min': np.min(timings),
            'max': np.max(timings),
            'std': np.std(timings),
            'total': np.sum(timings),
        }

    def print_summary(self) -> None:
        """Print performance summary to console."""
        print("\n" + "="*70)
        print("Performance Summary")
        print("="*70)

        for timer_name in sorted(self.timings.keys()):
            stats = self.get_statistics(timer_name)
            if stats:
                print(
                    f"{timer_name:30s} | "
                    f"count: {stats['count']:6d} | "
                    f"mean: {stats['mean']*1000:8.3f}ms | "
                    f"min: {stats['min']*1000:8.3f}ms | "
                    f"max: {stats['max']*1000:8.3f}ms"
                )

        print("="*70 + "\n")

    def save_csv(self) -> None:
        """Save timing statistics to CSV."""
        with open(self.csv_path, 'w') as f:
            f.write("Timer,Count,Mean(ms),Min(ms),Max(ms),Std(ms),Total(s)\n")
            for timer_name in sorted(self.timings.keys()):
                stats = self.get_statistics(timer_name)
                if stats:
                    f.write(
                        f"{timer_name},"
                        f"{stats['count']},"
                        f"{stats['mean']*1000:.3f},"
                        f"{stats['min']*1000:.3f},"
                        f"{stats['max']*1000:.3f},"
                        f"{stats['std']*1000:.3f},"
                        f"{stats['total']:.3f}\n"
                    )


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, logger: PerformanceLogger, name: str):
        """
        Initialize timer.

        Args:
            logger: PerformanceLogger instance
            name: Timer name
        """
        self.logger = logger
        self.name = name
        self.start_time = None

    def __enter__(self):
        """Enter context."""
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record timing."""
        import time
        duration = time.perf_counter() - self.start_time
        self.logger.record_timing(self.name, duration)


def setup_logging(log_dir: str = './logs', level: str = 'INFO') -> Dict[str, Any]:
    """
    Setup comprehensive logging infrastructure.

    Args:
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Dictionary containing all loggers
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Set root logger level
    logging.root.setLevel(getattr(logging, level))

    loggers = {
        'mission': MissionLogger('mission', log_dir),
        'metrics': MetricsLogger('metrics', log_dir),
        'performance': PerformanceLogger('performance', log_dir),
    }

    return loggers


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # Add handlers if none exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def configure_file_logging(logger_name: str, filepath: str,
                          level: str = 'DEBUG', json_format: bool = False) -> None:
    """
    Configure file logging for a logger.

    Args:
        logger_name: Name of logger
        filepath: Path to log file
        level: Logging level
        json_format: Use JSON format
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level))

    # Create file handler
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(getattr(logging, level))

    # Set formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def configure_rotating_file_logging(logger_name: str, filepath: str,
                                   max_bytes: int = 10485760,
                                   backup_count: int = 5) -> None:
    """
    Configure rotating file logging.

    Args:
        logger_name: Name of logger
        filepath: Path to log file
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
    """
    logger = logging.getLogger(logger_name)

    rotating_handler = logging.handlers.RotatingFileHandler(
        filepath, maxBytes=max_bytes, backupCount=backup_count
    )
    rotating_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    rotating_handler.setFormatter(formatter)
    logger.addHandler(rotating_handler)

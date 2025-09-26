import logging
import logging.handlers
from pathlib import Path
import json
from datetime import datetime
import sys
import os
import re
from typing import Optional, Dict, Any, List

class JSONFormatter(logging.Formatter):
    """Formata logs em JSON para fácil parsing e análise."""
    
    def __init__(self, include_extra_fields: bool = True):
        super().__init__()
        self.include_extra_fields = include_extra_fields
    
    def format(self, record):
        try:
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage(),
                'process': record.process,
                'thread': record.threadName,
            }
            
            # Contexto adicional
            if self.include_extra_fields and hasattr(record, 'extra'):
                log_data.update(record.extra)
                
            # Exceções
            if record.exc_info:
                log_data['exception'] = self.formatException(record.exc_info)
                
            return json.dumps(log_data, ensure_ascii=False, default=str)
            
        except Exception as e:
            return json.dumps({
                'timestamp': datetime.now().isoformat(),
                'level': 'ERROR',
                'logger': 'JSONFormatter',
                'message': f'Error formatting log: {str(e)}'
            })

class SensitiveDataFilter(logging.Filter):
    """Filtra dados sensíveis dos logs."""
    
    def __init__(self, patterns: Optional[List[str]] = None):
        super().__init__()
        self.patterns = patterns or [
            r'(password|api[_-]key|token|secret)=([^&\s]+)',
            r'([0-9]{4}-?){4}',  # Cartão de crédito básico
        ]
    
    def filter(self, record):
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            for pattern in self.patterns:
                msg = re.sub(pattern, r'\1=***', msg)
            record.msg = msg
        return True

def setup_logging(
    app_name: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Optional[Path] = None,
    console_format: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 7,
    filter_sensitive_data: bool = True
) -> logging.Logger:
    """
    Configura logging estruturado para o aplicativo.
    """
    # Configuração padrão
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove handlers existentes
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Formatter para console
    console_formatter = logging.Formatter(
        console_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    if filter_sensitive_data:
        console_handler.addFilter(SensitiveDataFilter())
    
    root_logger.addHandler(console_handler)
    
    if log_to_file:
        # JSON formatter para arquivos
        json_formatter = JSONFormatter()
        
        # Arquivo principal
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / f"{app_name}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(json_formatter)
        file_handler.addFilter(SensitiveDataFilter())
        
        # Arquivo de erros
        error_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / f"{app_name}_error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setFormatter(json_formatter)
        error_handler.setLevel(logging.ERROR)
        error_handler.addFilter(SensitiveDataFilter())
        
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
    
    # Configura bibliotecas externas
    for lib in ['urllib3', 'requests', 'ccxt', 'asyncio']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    return root_logger

# Exemplo de uso com contexto adicional
def get_trading_logger(name: str = "trading_bot"):
    logger = logging.getLogger(name)
    
    # Adiciona contexto padrão para logs de trading
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.extra = {
            'app': 'trading_bot',
            'environment': os.getenv('ENVIRONMENT', 'development')
        }
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    return logger

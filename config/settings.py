import os
from dataclasses import dataclass
from typing import Optional
import logging

@dataclass
class APIConfig:
    alpha_vantage_key: str
    news_api_key: str
    alpaca_api_key: str
    alpaca_secret_key: str
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None

@dataclass
class DatabaseConfig:
    url: str = "sqlite:///data/trading.db"
    redis_url: str = "redis://localhost:6379"

@dataclass
class TradingConfig:
    paper_trading: bool = True
    max_position_size: float = 0.2
    risk_per_trade: float = 0.02
    commission: float = 0.001

@dataclass
class AlertConfig:
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_address: Optional[str] = None
    email_password: Optional[str] = None

@dataclass
class AppConfig:
    debug: bool = False
    log_level: str = "INFO"
    port: int = 8

501
    host: str = "0.0.0.0"

class Settings:
    """Application settings loaded from environment variables"""
    
    def __init__(self):
        self.api = APIConfig(
            alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
            news_api_key=os.getenv('NEWS_API_KEY'),
            alpaca_api_key=os.getenv('ALPACA_API_KEY'),
            alpaca_secret_key=os.getenv('ALPACA_SECRET_KEY'),
            reddit_client_id=os.getenv('REDDIT_CLIENT_ID'),
            reddit_client_secret=os.getenv('REDDIT_CLIENT_SECRET')
        )
        
        self.database = DatabaseConfig(
            url=os.getenv('DATABASE_URL', 'sqlite:///data/trading.db'),
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379')
        )
        
        self.trading = TradingConfig(
            paper_trading=os.getenv('PAPER_TRADING', 'true').lower() == 'true',
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', '0.2')),
            risk_per_trade=float(os.getenv('RISK_PER_TRADE', '0.02')),
            commission=float(os.getenv('COMMISSION', '0.001'))
        )
        
        self.alerts = AlertConfig(
            telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID'),
            smtp_server=os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            smtp_port=int(os.getenv('SMTP_PORT', '587')),
            email_address=os.getenv('EMAIL_ADDRESS'),
            email_password=os.getenv('EMAIL_PASSWORD')
        )
        
        self.app = AppConfig(
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            port=int(os.getenv('PORT', '8501')),
            host=os.getenv('HOST', '0.0.0.0')
        )
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging based on settings"""
        logging.basicConfig(
            level=getattr(logging, self.app.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/logs/trading.log'),
                logging.StreamHandler()
            ]
        )
    
    def validate(self) -> list:
        """Validate configuration and return list of errors"""
        errors = []
        
        if not self.api.alpha_vantage_key:
            errors.append("ALPHA_VANTAGE_API_KEY is required")
        
        if not self.api.alpaca_api_key:
            errors.append("ALPACA_API_KEY is required")
        
        if not self.api.alpaca_secret_key:
            errors.append("ALPACA_SECRET_KEY is required")
        
        if self.trading.max_position_size <= 0 or self.trading.max_position_size > 1:
            errors.append("MAX_POSITION_SIZE must be between 0 and 1")
        
        return errors

# Global settings instance
settings = Settings()
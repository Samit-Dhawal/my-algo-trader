import smtplib
import logging
import asyncio
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from dataclasses import dataclass
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class AlertMessage:
    title: str
    message: str
    alert_type: str  # 'trade', 'signal', 'error', 'info'
    symbol: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = None

class TelegramBot:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Telegram message sent successfully")
                        return True
                    else:
                        logger.error(f"Telegram API error: {await response.text()}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def format_trade_alert(self, alert: AlertMessage) -> str:
        """Format trade alert for Telegram"""
        emoji_map = {
            'trade': '💰',
            'signal': '📊',
            'error': '❌',
            'info': 'ℹ️'
        }
        
        emoji = emoji_map.get(alert.alert_type, 'ℹ️')
        
        message = f"{emoji} <b>{alert.title}</b>\n\n"
        message += f"{alert.message}\n"
        
        if alert.symbol:
            message += f"Symbol: <code>{alert.symbol}</code>\n"
        if alert.price:
            message += f"Price: <code>${alert.price:.4f}</code>\n"
        if alert.quantity:
            message += f"Quantity: <code>{alert.quantity}</code>\n"
        
        message += f"\n🕐 {self._get_timestamp()}"
        
        return message
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class EmailAlert:
    def __init__(self, smtp_server: str, smtp_port: int, email: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password
    
    def send_email(self, to_email: str, alert: AlertMessage) -> bool:
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = to_email
            msg['Subject'] = f"Trading Alert: {alert.title}"
            
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _format_email_body(self, alert: AlertMessage) -> str:
        """Format email body with HTML"""
        html = f"""
        <html>
        <body>
            <h2 style="color: #333;">{alert.title}</h2>
            <p>{alert.message}</p>
            
            {f'<p><strong>Symbol:</strong> {alert.symbol}</p>' if alert.symbol else ''}
            {f'<p><strong>Price:</strong> ${alert.price:.4f}</p>' if alert.price else ''}
            {f'<p><strong>Quantity:</strong> {alert.quantity}</p>' if alert.quantity else ''}
            
            <p style="color: #666; font-size: 12px;">
                Alert Type: {alert.alert_type.title()}<br>
                Timestamp: {self._get_timestamp()}
            </p>
            
            <hr>
            <p style="color: #999; font-size: 10px;">
                This is an automated message from your Trading System
            </p>
        </body>
        </html>
        """
        return html
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

class AlertSystem:
    def __init__(self):
        self.telegram_bot = None
        self.email_alert = None
        
        # Initialize Telegram bot if configured
        if settings.alerts.telegram_bot_token and settings.alerts.telegram_chat_id:
            self.telegram_bot = TelegramBot(
                settings.alerts.telegram_bot_token,
                settings.alerts.telegram_chat_id
            )
        
        # Initialize email alerts if configured
        if all([settings.alerts.email_address, settings.alerts.email_password]):
            self.email_alert = EmailAlert(
                settings.alerts.smtp_server,
                settings.alerts.smtp_port,
                settings.alerts.email_address,
                settings.alerts.email_password
            )
    
    async def send_alert(self, alert: AlertMessage) -> Dict[str, bool]:
        """Send alert via all configured channels"""
        results = {}
        
        # Send Telegram alert
        if self.telegram_bot:
            message = self.telegram_bot.format_trade_alert(alert)
            results['telegram'] = await self.telegram_bot.send_message(message)
        
        # Send email alert
        if self.email_alert and settings.alerts.email_address:
            results['email'] = self.email_alert.send_email(
                settings.alerts.email_address, alert
            )
        
        return results
    
    def send_trade_alert(self, symbol: str, side: str, quantity: int, price: float, status: str):
        """Send trade execution alert"""
        alert = AlertMessage(
            title=f"Trade {status.title()}",
            message=f"Trade executed: {side.upper()} {quantity} shares of {symbol}",
            alert_type='trade',
            symbol=symbol,
            price=price,
            quantity=quantity
        )
        
        # Run async alert sending
        asyncio.create_task(self.send_alert(alert))
    
    def send_signal_alert(self, symbol: str, signal: int, confidence: float, method: str):
        """Send trading signal alert"""
        signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
        
        alert = AlertMessage(
            title=f"Trading Signal: {signal_text}",
            message=f"New {signal_text} signal generated for {symbol} using {method}",
            alert_type='signal',
            symbol=symbol
        )
        alert.message += f"\nConfidence: {confidence:.3f}"
        
        asyncio.create_task(self.send_alert(alert))
    
    def send_error_alert(self, error_message: str, component: str = "System"):
        """Send error alert"""
        alert = AlertMessage(
            title=f"Trading System Error",
            message=f"Error in {component}: {error_message}",
            alert_type='error'
        )
        
        asyncio.create_task(self.send_alert(alert))
    
    def send_system_status(self, status: str, details: str = ""):
        """Send system status alert"""
        alert = AlertMessage(
            title="System Status Update",
            message=f"Status: {status}\n{details}",
            alert_type='info'
        )
        
        asyncio.create_task(self.send_alert(alert))

# Global alert system instance
alert_system = AlertSystem()
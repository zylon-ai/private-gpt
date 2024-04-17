import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from private_gpt.users.core.config import settings
from fastapi import HTTPException, status

def send_registration_email(fullname: str, email: str, random_password: str) -> None:
    """
    Send a registration email with a random password.
    """
    subject = "Welcome to QuickGPT - Registration Successful"
    body = f"""
        <html>
        <body>
            <p>Hello {fullname},</p>
            
            <p>Thank you for registering with QuickGPT!</p>
            
            <p>Your temporary password is: <strong>{random_password}</strong></p>
            
            <p>Please log in to QuickGPT <a href="http://quickgpt.gibl.com.np">here</a>.</p>
            
            <p>Please use this password to log in and consider changing it to a more secure one after logging in.</p>
            
            <p>Best regards,<br>
            QuickGPT Team</p>
        </body>
        </html>
    """

    msg = MIMEMultipart()
    msg.attach(MIMEText(body, "html"))
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_SENDER_EMAIL
    msg["To"] = email

    print(settings.SMTP_SERVER)
    print(settings.SMTP_PORT)
    
    try:
        with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
            # server.starttls()
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_SENDER_EMAIL, email, msg.as_string())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to send email."
        )
    

def validate_password(password: str) -> None:
    """
    Validate the password according to the defined criteria.
    
    Args:
        password (str): The new password to validate.
        
    Raises:
        HTTPException: If the password does not meet the criteria.
    """
    # Define the password validation criteria
    min_length = 6
    require_upper = re.compile(r'[A-Z]')
    require_lower = re.compile(r'[a-z]')
    require_digit = re.compile(r'\d')
    require_special = re.compile(r'[!@#$%^&*()_+=-]')  # Add special characters as needed

    # Check password length
    if len(password) < min_length:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Password must be at least {min_length} characters long.")

    # Check for uppercase letter
    if not require_upper.search(password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must contain at least one uppercase letter.")

    # Check for lowercase letter
    if not require_lower.search(password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must contain at least one lowercase letter.")

    # Check for digit
    if not require_digit.search(password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must contain at least one digit.")

    # Check for special character
    if not require_special.search(password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must contain at least one special character (e.g., !@#$%^&*()_+=-).")

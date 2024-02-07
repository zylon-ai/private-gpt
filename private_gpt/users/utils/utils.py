import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from private_gpt.users.core.config import settings


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
    msg.attach(MIMEText(body, "plain"))
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_SENDER_EMAIL
    msg["To"] = email

    print(settings.SMTP_SERVER)
    print(settings.SMTP_PORT)

    with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
        server.starttls()
        server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
        server.sendmail(settings.SMTP_SENDER_EMAIL, email, msg.as_string())


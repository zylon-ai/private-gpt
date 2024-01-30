import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from private_gpt.users.core.config import settings


def send_registration_email(fullname: str, email: str, random_password: str) -> None:
    """
    Send a registration email with a random password.
    """
    subject = "Welcome to QuickGPT - Registration Successful"
    body = f"Hello {fullname},\n\nThank you for registering with QuickGPT!\n\n"\
       f"Your temporary password is: {random_password}\n"\
       f"Please login to the QuickGPT: http://quickgpt.gibl.com.np\n"\
       f"Please use this password to log in and consider changing it"\
       " to a more secure one after logging in.\n\n"\
       "Best regards,\nQuickGPT Team"

    msg = MIMEMultipart()
    msg.attach(MIMEText(body, "plain"))
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_SENDER_EMAIL
    msg["To"] = email

    with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
        server.starttls()
        server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
        server.sendmail(settings.SMTP_SENDER_EMAIL, email, msg.as_string())


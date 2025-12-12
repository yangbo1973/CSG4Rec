import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header


class EmailSender:
    def __init__(self, sender_email, auth_code, smtp_server="smtp.qq.com", smtp_port=465):
        """
        :param sender_email: sender email address (e.g. your QQ email)
        :param auth_code: SMTP authorization code (not the account password)
        :param smtp_server: SMTP server address (default: QQ SMTP)
        :param smtp_port: SMTP SSL port (QQ uses 465)
        """
        self.sender_email = sender_email
        self.auth_code = auth_code
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send_email(self, to_email, subject, content, attachments=None):
        """
        Send an email.
        :param to_email: recipient email address
        :param subject: email subject
        :param content: email body text
        :param attachments: optional list of file paths to attach
        """
        try:
            # Build the message
            msg = MIMEMultipart()
            msg['From'] = Header(self.sender_email)
            msg['To'] = Header(to_email)
            msg['Subject'] = Header(subject, 'utf-8')

            # Attach body
            msg.attach(MIMEText(content, 'plain', 'utf-8'))

            # Attach files if provided
            if attachments:
                for file in attachments:
                    with open(file, 'rb') as f:
                        attach = MIMEText(f.read(), 'base64', 'utf-8')
                        attach["Content-Type"] = 'application/octet-stream'
                        attach["Content-Disposition"] = f'attachment; filename="{os.path.basename(file)}"'
                        msg.attach(attach)

            # Send the email over SSL
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.sender_email, self.auth_code)
                server.sendmail(self.sender_email, [to_email], msg.as_string())

            print("✅ Email sent successfully!")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")


if __name__ == "__main__":
    # ===== Example usage =====
    # Replace the placeholders below with your own values or load them from environment variables.
    sender = EmailSender(
        sender_email=os.getenv("SENDER_EMAIL", "your_email@example.com"),
        auth_code=os.getenv("SMTP_AUTH_CODE", "your_smtp_auth_code")  # obtain from your email provider
    )

    sender.send_email(
        to_email="recipient@example.com",
        subject="Experiment Results Notification",
        content="The experiment has completed. Results:\n - HR@10 = 0.435\n - NDCG@10 = 0.271",
        # attachments=["./result.json"]  # optional: list of file paths
    )

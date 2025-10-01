#!/usr/bin/env python3
import os
import smtplib
from email.message import EmailMessage
import sys

def main():
    recipient = os.environ.get("PIPELINE_EMAIL")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    tests_status = os.environ.get("TESTS_STATUS", "unknown")
    build_status = os.environ.get("BUILD_STATUS", "unknown")

    if not recipient or not smtp_user or not smtp_pass:
        print("Missing PIPELINE_EMAIL or SMTP_USER or SMTP_PASS", file=sys.stderr)
        sys.exit(2)

    subject = f"CI Pipeline — tests: {tests_status} — build: {build_status}"
    body = f"Pipeline executado.\n\nResultados:\n- Tests: {tests_status}\n- Build: {build_status}\n\nAbra o Actions para ver detalhes e artefatos."

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port) as s:
        s.starttls()
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)

    print("Notification sent to", recipient)

if __name__ == "__main__":
    main()

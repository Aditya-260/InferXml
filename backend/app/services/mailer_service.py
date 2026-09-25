"""
Mailer bridge for OTP and reset emails.
"""
import json
import os
from urllib import error, request


class MailerError(Exception):
    """Raised when the mailer service cannot send an email."""


class MailerService:
    """Thin HTTP client for the Node mailer service."""

    def __init__(self):
        self.base_url = os.getenv('MAILER_URL', 'http://mailer:4000')

    def send_template_email(self, template_type, to_email, payload):
        body = json.dumps({
            'type': template_type,
            'to': to_email,
            'payload': payload,
        }).encode('utf-8')

        req = request.Request(
            f'{self.base_url}/send',
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        try:
            with request.urlopen(req, timeout=10) as response:
                if response.status >= 400:
                    raise MailerError('Mailer service returned an error response')
        except error.HTTPError as exc:
            raise MailerError(f'Mailer service error: {exc.code}') from exc
        except error.URLError as exc:
            raise MailerError('Mailer service unavailable') from exc


mailer_service = MailerService()

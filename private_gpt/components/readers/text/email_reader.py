import contextlib
import datetime
import logging
import re
from collections.abc import Iterator
from email.message import Message
from email.parser import BytesParser
from email.policy import default
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Any

import html2text
from llama_index.core.schema import BaseNode, Document
from pydantic import BaseModel, Field

from private_gpt.components.readers.text.text_reader import TextReader
from private_gpt.settings.settings import settings

debug_mode = settings().server.debug_mode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)


class EmailAddress(BaseModel):
    """Model for email address with optional display name."""

    name: str = ""
    address: str = ""

    def __str__(self) -> str:
        if self.name and self.address:
            return f"<a href='mailto:{self.address}'>{self.name}</a>"
        elif self.address:
            return f"<a href='mailto:{self.address}'>{self.address}</a>"
        return self.name or self.address


class EmailContent(BaseModel):
    """Structured representation of an email for LLM consumption."""

    subject: str = ""
    from_: list[EmailAddress] = Field(default_factory=list)
    to: list[EmailAddress] = Field(default_factory=list)
    cc: list[EmailAddress] = Field(default_factory=list)
    bcc: list[EmailAddress] = Field(default_factory=list)
    date: datetime.datetime | None = None
    body_text: str = ""
    body_html: str = ""
    attachments: list[dict[str, Any]] = Field(default_factory=list)

    def to_llm_friendly_text(self) -> str:
        """Convert the email to a format that is easier for LLMs to understand."""
        parts = []

        # Email header as HTML
        parts.append(f"<h2>Email: {self.subject}</h2>")
        parts.append("<div class='email-header'>")

        # Format from addresses
        from_str = ", ".join(str(email) for email in self.from_)
        parts.append(f"<p>From: {from_str}</p>")

        # Format to addresses
        to_str = ", ".join(str(email) for email in self.to)
        parts.append(f"<p>To: {to_str}</p>")

        # Format CC addresses
        if self.cc:
            cc_str = ", ".join(str(email) for email in self.cc)
            parts.append(f"<p>CC: {cc_str}</p>")

        # Format BCC addresses
        if self.bcc:
            bcc_str = ", ".join(str(email) for email in self.bcc)
            parts.append(f"<p>BCC: {bcc_str}</p>")

        if self.date:
            parts.append(f"<p>Date: {self.date.strftime('%Y-%m-%d %H:%M:%S')}</p>")

        parts.append("</div>")
        parts.append("<div class='email-body'>")

        # Format email body as HTML
        if self.body_text:
            # Convert plain text to HTML (preserving line breaks)
            body_html = self.body_text.replace("\n", "<br/>")
            parts.append(f"<p>{body_html}</p>")
        elif self.body_html:
            parts.append(self.body_html)

        parts.append("</div>")

        if self.attachments:
            parts.append("<p>Attachments</p>")
            parts.append("<ul>")
            for attachment in self.attachments:
                size_str = ""
                if "size" in attachment:
                    size_kb = attachment["size"] / 1024
                    if size_kb > 1024:
                        size_str = f" ({size_kb / 1024:.1f} MB)"
                    else:
                        size_str = f" ({size_kb:.1f} KB)"

                parts.append(
                    f"<li>{attachment['filename']} — {attachment['content_type']}{size_str}</li>"
                )
            parts.append("</ul>")

        try:
            from markdownify import markdownify as md

            markdown: str = md(
                "\n".join(parts),
                heading_style="ATX",
                bullets="bullet_list",
                bullets_strong=True,
            )
            return markdown
        except ImportError:
            return "\n".join(parts)

    def get_extracted_emails(self) -> dict[str, list[str]]:
        """Extract just the email addresses from all fields.

        Returns:
            Dict with keys 'from', 'to', 'cc', 'bcc' containing lists of email addresses
        """
        return {
            "from": [email.address for email in self.from_],
            "to": [email.address for email in self.to],
            "cc": [email.address for email in self.cc],
            "bcc": [email.address for email in self.bcc],
        }


class EmailTextReader(TextReader):
    """Reader for handling emails."""

    include_attachments: bool
    html_to_text: bool
    html_converter: html2text.HTML2Text

    _last_parsed_email: EmailContent | None = None

    def __init__(self, include_attachments: bool = False, html_to_text: bool = True):
        """Initialize the email reader."""
        html_converter = html2text.HTML2Text()
        html_converter.ignore_links = False
        html_converter.ignore_images = False
        html_converter.ignore_tables = False

        super().__init__(
            include_attachments=include_attachments,
            html_to_text=html_to_text,
            html_converter=html_converter,
        )

    def lazy_document_load(
        self,
        file_path: Path,
        encoding: str | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> Iterator[BaseNode]:
        """Load email file and convert to an LLM-friendly format."""
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=default).parse(f)

        email_content = self._process_email(msg)
        self._last_parsed_email = email_content

        # Add extracted emails to extra_info
        if extra_info is None:
            extra_info = {}

        extra_info["extracted_emails"] = email_content.get_extracted_emails()

        yield Document(
            text=email_content.to_llm_friendly_text(),
            extra_info=extra_info,
        )

    def _process_email(self, msg: Message) -> EmailContent:
        """Process an email message into structured content."""
        email_content = EmailContent()

        email_content.subject = msg.get("Subject", "")

        # Process From field
        from_field = msg.get("From", "")
        if from_field:
            email_content.from_ = self._parse_email_addresses(from_field)

        # Process To field
        to_field = msg.get("To", "")
        if to_field:
            email_content.to = self._parse_email_addresses(to_field)

        # Process CC field
        cc_field = msg.get("Cc", "")
        if cc_field:
            email_content.cc = self._parse_email_addresses(cc_field)

        # Process BCC field
        bcc_field = msg.get("Bcc", "")
        if bcc_field:
            email_content.bcc = self._parse_email_addresses(bcc_field)

        # Process date
        date_str = msg.get("Date", "")
        if date_str:
            with contextlib.suppress(Exception):
                email_content.date = parsedate_to_datetime(date_str)

        self._extract_content_and_attachments(msg, email_content)

        return email_content

    @staticmethod
    def _parse_email_addresses(address_field: str) -> list[EmailAddress]:
        """Parse email addresses from header fields.

        Args:
            address_field: Raw address field string like "Name <email@example.com>"

        Returns:
            List of EmailAddress objects
        """
        parsed_addresses = []
        # Parse addresses using email.utils.getaddresses
        addresses = getaddresses([address_field])

        for name, addr in addresses:
            if addr:  # Only add if there's an actual email address
                parsed_addresses.append(EmailAddress(name=name, address=addr))

        return parsed_addresses

    def _extract_content_and_attachments(
        self, msg: Message, email_content: EmailContent
    ) -> None:
        """Extract body content and attachments from email."""
        is_multipart = msg.is_multipart()
        processed_plain_text = False
        processed_html = False

        if is_multipart:
            main_parts = []
            for part in msg.get_payload():
                if not isinstance(part, Message):
                    continue

                if part.get_content_type() == "text/plain" and not processed_plain_text:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload and isinstance(payload, bytes):
                            charset = part.get_content_charset() or "utf-8"
                            text = payload.decode(charset, errors="replace")
                            text = self.clean_email_text(text)
                            if text:
                                email_content.body_text = text
                                processed_plain_text = True
                    except Exception as e:
                        logger.warning(f"Error extracting plain text: {e!s}")

                elif (
                    part.get_content_type() == "text/html"
                    and not processed_html
                    and not processed_plain_text
                ):
                    try:
                        payload = part.get_payload(decode=True)
                        if payload and isinstance(payload, bytes):
                            charset = part.get_content_charset() or "utf-8"
                            html = payload.decode(charset, errors="replace")
                            if html:
                                if self.html_to_text:
                                    text = self.html_converter.handle(html)
                                    text = self.clean_email_text(text)
                                    if text:
                                        email_content.body_text = text
                                        processed_plain_text = True
                                else:
                                    email_content.body_html = html
                                    processed_html = True
                    except Exception as e:
                        logger.warning(f"Error extracting HTML content: {e!s}")

                elif part.is_multipart():
                    for subpart in part.get_payload():
                        if isinstance(subpart, Message):
                            main_parts.append(subpart)

                elif (
                    part.get_content_disposition() == "attachment"
                    and self.include_attachments
                ):
                    filename = part.get_filename()
                    if filename:
                        payload = part.get_payload(decode=True)
                        size = len(payload) if payload else 0
                        email_content.attachments.append(
                            {
                                "filename": filename,
                                "content_type": part.get_content_type(),
                                "size": size,
                            }
                        )
                elif part.get_content_disposition() != "inline":
                    main_parts.append(part)

            # Process any nested parts
            for part in main_parts:
                if not isinstance(part, Message):
                    continue

                if part.get_content_type() == "text/plain" and not processed_plain_text:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload and isinstance(payload, bytes):
                            charset = part.get_content_charset() or "utf-8"
                            text = payload.decode(charset, errors="replace")
                            text = self.clean_email_text(text)
                            if text:
                                email_content.body_text = text
                                processed_plain_text = True
                    except Exception as e:
                        logger.warning(f"Error extracting plain text: {e!s}")

                elif (
                    part.get_content_type() == "text/html"
                    and not processed_html
                    and not processed_plain_text
                ):
                    try:
                        payload = part.get_payload(decode=True)
                        if payload and isinstance(payload, bytes):
                            charset = part.get_content_charset() or "utf-8"
                            html = payload.decode(charset, errors="replace")
                            if html:
                                if self.html_to_text:
                                    text = self.html_converter.handle(html)
                                    text = self.clean_email_text(text)
                                    if text:
                                        email_content.body_text = text
                                        processed_plain_text = True
                                else:
                                    email_content.body_html = html
                                    processed_html = True
                    except Exception as e:
                        logger.warning(f"Error extracting HTML content: {e!s}")

                elif (
                    part.get_content_disposition() == "attachment"
                    and self.include_attachments
                ):
                    filename = part.get_filename()
                    if filename:
                        payload = part.get_payload(decode=True)
                        size = len(payload) if payload else 0
                        email_content.attachments.append(
                            {
                                "filename": filename,
                                "content_type": part.get_content_type(),
                                "size": size,
                            }
                        )
        else:
            # Handle non-multipart messages
            payload = msg.get_payload(decode=True)
            if payload and isinstance(payload, bytes):
                try:
                    charset = msg.get_content_charset() or "utf-8"
                    text = payload.decode(charset, errors="replace")
                    text = self.clean_email_text(text)
                    email_content.body_text = text
                except Exception as e:
                    logger.warning(f"Error decoding email body: {e!s}")

        if not email_content.body_text and not email_content.body_html:
            logger.warning("No email body content was extracted")

    @staticmethod
    def clean_email_text(text: str) -> str:
        """Clean email text by removing excessive whitespace and common noise."""
        if not text:
            return ""

        # Remove email forwarding/reply markers
        text = re.sub(r"^>+\s*", "", text, flags=re.MULTILINE)

        # Remove excessive newlines (more than 2 in a row)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Preserve signatures but remove mobile signatures
        signatures = [
            r"Sent from my .*",  # Mobile signatures
        ]

        for pattern in signatures:
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

        return text.strip()

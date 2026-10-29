from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import default

from private_gpt.components.readers.text.email_reader import EmailTextReader


def _body(message: EmailMessage, **reader_kwargs: bool) -> tuple[str, list[str]]:
    parsed = BytesParser(policy=default).parsebytes(message.as_bytes())
    content = EmailTextReader(**reader_kwargs)._process_email(parsed)
    return content.body_text, [a["filename"] for a in content.attachments]


def _message() -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = "Test"
    message["From"] = "sender@example.com"
    return message


def test_email_reader_converts_a_single_part_html_message() -> None:
    message = _message()
    message.set_content("<p>Hello <b>newsletter</b> reader.</p>", subtype="html")

    body, _ = _body(message)

    assert body == "Hello **newsletter** reader."


def test_email_reader_finds_a_body_nested_three_levels_down() -> None:
    # mixed > related > alternative: a body with an inline logo, plus an attachment
    alternative = EmailMessage()
    alternative.set_content("Body three levels deep.")
    alternative.add_alternative("<p>Body three levels deep.</p>", subtype="html")
    related = EmailMessage()
    related.make_related()
    related.attach(alternative)
    logo = EmailMessage()
    logo.set_content(b"\x89PNG", maintype="image", subtype="png", cid="<logo>")
    related.attach(logo)
    message = _message()
    message.make_mixed()
    message.attach(related)
    message.add_attachment(
        b"%PDF-1.4", maintype="application", subtype="pdf", filename="report.pdf"
    )

    body, attachments = _body(message, include_attachments=True)

    assert body == "Body three levels deep."
    assert attachments == ["report.pdf"]


def test_email_reader_keeps_both_halves_of_a_body_split_by_an_inline_image() -> None:
    # Apple Mail writes the text before and after an inline image as two parts
    message = _message()
    message.make_mixed()
    for text in ("First half of the message.", None, "Second half after the image."):
        part = EmailMessage()
        if text is None:
            part.set_content(
                b"\x89PNG",
                maintype="image",
                subtype="png",
                disposition="inline",
                filename="a.png",
            )
        else:
            part.set_content(text)
            part["Content-Disposition"] = "inline"
        message.attach(part)

    body, _ = _body(message)

    assert body == "First half of the message.\n\nSecond half after the image."

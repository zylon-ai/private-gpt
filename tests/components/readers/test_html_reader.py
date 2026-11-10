from private_gpt.components.readers.text.html_reader import HtmlReader


def test_html_reader_keeps_the_text_of_namespaced_tags() -> None:
    # Inline XBRL (SEC filings) keeps every reported figure in an ix: tag, and
    # Word's HTML export wraps place names in st1: smart tags; the hidden
    # ix:header stays out.
    html = """<html><body>
<div style="display:none"><ix:header><ix:hidden>hidden facts</ix:hidden></ix:header></div>
<p>Net revenue was <ix:nonFraction name="us-gaap:Revenues">1,234</ix:nonFraction> million.</p>
<table><tr><td>Revenue</td><td><ix:nonFraction name="us-gaap:Revenues">1,234</ix:nonFraction></td></tr></table>
<ix:nonNumeric name="dei:DocumentType"><p>Annual report text block.</p></ix:nonNumeric>
<p>I live in <st1:City><st1:place>Berlin</st1:place></st1:City>.<o:p></o:p></p>
</body></html>"""

    markdown = HtmlReader()._process_content(html)

    assert "Net revenue was 1,234 million." in markdown
    assert "| Revenue | 1,234 |" in markdown
    assert "Annual report text block." in markdown
    assert "I live in Berlin." in markdown
    assert "hidden facts" not in markdown

from textwrap import dedent

from private_gpt.components.engines.citations.term_extractor import TextAnalyzer


def test_english_financial_table() -> None:
    text = dedent(
        """
    | Revenue | 2023 | 2022 |
    |---------|------|------|
    | Sales   | 100  | 90   |
    | Costs   | 70   | 65   |
    | Profit  | 30   | 25   |
    """
    )

    analyzer = TextAnalyzer()
    terms = analyzer.get_terms(text, min_length=4, max_length=6)

    expected = {"sale", "cost", "profit"}
    assert expected.issubset(terms), f"Expected {expected}, got {terms}"


def test_spanish_financial_table() -> None:
    text = dedent(
        """
    | Ingresos | 2023 | 2022 |
    |----------|-------|------|
    | Ventas   | 100   | 90   |
    | Costes   | 70    | 65   |
    | Beneficio| 30    | 25   |
    """
    )

    analyzer = TextAnalyzer()
    terms = analyzer.get_terms(text)

    expected = {"ventas", "costes"}
    assert expected.issubset(terms), f"Expected {expected}, got {terms}"


def test_numbered_list() -> None:
    text = dedent(
        """
    1. First quarter results
    2. Second quarter forecast
    3. Third quarter targets
        a. Sales goals
        b. Cost reduction
    """
    )

    analyzer = TextAnalyzer()
    terms = analyzer.get_terms(text, max_length=8)

    expected = {"quarter", "result", "forecast", "target", "sale", "goal"}
    assert expected.issubset(terms)


def test_mixed_content_extraction() -> None:
    text = dedent(
        """
    # Financial Results 2023

    Key highlights of our performance:

    | Metric   | Value |
    |----------|-------|
    | Revenue  | 100M  |
    | Profit   | 30M   |

    Notable achievements include revenue growth
    and profit improvement.
    """
    )

    analyzer = TextAnalyzer()
    terms = analyzer.get_terms(text, min_length=4)

    expected = {"revenue", "profit", "growth", "metric"}
    assert expected.issubset(terms)


def test_empty_content() -> None:
    analyzer = TextAnalyzer()

    assert len(analyzer.get_terms("")) == 0
    assert len(analyzer.get_terms(" ")) == 0
    assert len(analyzer.get_terms("\n")) == 0


def test_numeric_content() -> None:
    text = "100 200 300 revenue2023 profit2022 2024forecast"
    analyzer = TextAnalyzer()
    terms = analyzer.get_terms(text)

    assert "100" not in terms
    assert "revenue2023" not in terms
    assert "profit2022" not in terms
    assert "2024forecast" not in terms


def test_only_numeric_content() -> None:
    text = "100 200 300 2023 2024 20225"
    analyzer = TextAnalyzer()
    terms = analyzer.get_terms(text)
    assert len(terms) == 0


def test_length_constraints_none() -> None:
    text = "cat table elephant supercalifragilistic"
    analyzer = TextAnalyzer()
    terms = analyzer.get_terms(text, min_length=None, max_length=None)

    expected = {"cat", "table", "elephant", "supercalifragilistic"}
    assert expected.issubset(terms)


def test_length_constraints_combinations() -> None:
    text = "cat table elephant supercalifragilistic"
    test_cases = [
        (3, None, {"cat", "table", "elephant", "supercalifragilistic"}),
        (None, 6, {"cat", "table"}),
        (4, 8, {"table", "elephant"}),
    ]

    for min_len, max_len, expected in test_cases:
        analyzer = TextAnalyzer()
        terms = analyzer.get_terms(text, min_length=min_len, max_length=max_len)
        assert expected.issubset(terms)


def test_multilingual_content() -> None:
    text = dedent(
        """
    # Financial Report / Informe Financiero

    | English | Español |
    |---------|---------|
    | Revenue | Ingresos |
    | Profit  | Beneficio |
    | Costs   | Costes  |
    """
    )

    analyzer = TextAnalyzer()
    terms = analyzer.get_terms(text)

    expected = {"revenue", "profit", "cost", "ingresos", "beneficio", "costes"}
    assert expected.issubset(terms)


def test_unique_terms_across_texts() -> None:
    texts = [
        dedent(
            """
        Revenue and profit analysis
        Key metrics review
        """
        ),
        dedent(
            """
        Revenue and cost breakdown
        Detailed analysis
        """
        ),
        dedent(
            """
        Detailed profit margins
        Performance metrics
        """
        ),
    ]

    analyzer = TextAnalyzer()
    unique_terms = analyzer.get_unique_terms(texts)

    # Verify uniqueness
    assert not any(
        "revenue" in terms for terms in unique_terms
    ), "Revenue appears in multiple texts"
    assert any("cost" in terms for terms in unique_terms), "Cost should be unique"
    assert any("margin" in terms for terms in unique_terms), "Margins should be unique"


def test_unique_terms_simple_case() -> None:
    texts = ["revenue profit margin", "revenue cost expense", "revenue sales growth"]

    analyzer = TextAnalyzer()
    unique_terms = analyzer.get_unique_terms(texts)

    assert set(unique_terms[0]) == {"profit", "margin"}
    assert set(unique_terms[1]) == {"cost", "expense"}
    assert set(unique_terms[2]) == {"sale", "growth"}


def test_unique_terms_with_overlap() -> None:
    texts = [
        "revenue profit margin sales",
        "revenue profit cost sales",
        "revenue profit margin growth",
    ]

    analyzer = TextAnalyzer()
    unique_terms = analyzer.get_unique_terms(texts)

    # Only truly unique terms should be included
    assert "revenue" not in [term for terms in unique_terms for term in terms]
    assert "profit" not in [term for terms in unique_terms for term in terms]
    assert "cost" in unique_terms[1]
    assert "growth" in unique_terms[2]


def test_unique_terms_with_case_variations() -> None:
    texts = ["Revenue Profit margin", "revenue PROFIT costs", "REVENUE profit GROWTH"]

    analyzer = TextAnalyzer()
    unique_terms = analyzer.get_unique_terms(texts)

    # Case variations should be treated as the same term
    assert "revenue" not in [term for terms in unique_terms for term in terms]
    assert "profit" not in [term for terms in unique_terms for term in terms]
    assert "cost" in unique_terms[1]
    assert "growth" in unique_terms[2]


def test_unique_terms_with_empty_text() -> None:
    texts = ["revenue profit", "", "revenue growth"]

    analyzer = TextAnalyzer()
    unique_terms = analyzer.get_unique_terms(texts)

    assert len(unique_terms[1]) == 0  # Empty text should yield no unique terms
    assert "profit" in unique_terms[0]
    assert "growth" in unique_terms[2]


def test_unique_terms_with_numbers() -> None:
    texts = ["revenue2023 profit", "revenue2024 costs", "revenue growth2025"]

    analyzer = TextAnalyzer()
    unique_terms = analyzer.get_unique_terms(texts)

    assert "profit" in unique_terms[0]
    assert "cost" in unique_terms[1]


def test_unique_terms_with_special_characters() -> None:
    texts = ["revenue-profit margin", "revenue/cost analysis", "revenue&growth metrics"]

    analyzer = TextAnalyzer()
    unique_terms = analyzer.get_unique_terms(texts)

    assert "margin" in unique_terms[0]
    assert "analysis" in unique_terms[1]
    assert "growth" in unique_terms[2]


def test_unique_terms_scoring_priority() -> None:
    texts = [
        dedent(
            """
        # Profit Analysis
        The profit margin was good
        Some profit details
        """
        ),
        dedent(
            """
        # Cost Review
        | Cost | Value |
        Other cost information
        """
        ),
    ]

    analyzer = TextAnalyzer()
    unique_terms = analyzer.get_unique_terms(texts)

    # Terms in headers should be scored higher and appear first
    if unique_terms[0]:  # If there are unique terms in first text
        assert unique_terms[0][0] == "profit"
    if unique_terms[1]:  # If there are unique terms in second text
        assert unique_terms[1][0] == "cost"


def test_unique_terms_length_constraints() -> None:
    texts = ["cat keyboard elephant", "dog chair giraffe", "rat house hippopotamus"]

    # Test with min_length
    analyzer = TextAnalyzer()
    unique_terms = analyzer.get_unique_terms(texts, min_length=5)
    assert "keyboard" in unique_terms[0]
    assert "cat" not in unique_terms[0]

    # Test with max_length
    analyzer = TextAnalyzer()
    unique_terms = analyzer.get_unique_terms(texts, max_length=5)
    assert "keyboard" not in unique_terms[0]
    assert "cat" in unique_terms[0]


def test_unique_terms_multilingual() -> None:
    texts = [
        "revenue profit margin",
        "ingresos beneficio costo",
        "revenue ingresos growth",
    ]

    analyzer = TextAnalyzer()
    unique_terms = analyzer.get_unique_terms(texts)

    assert "profit" in unique_terms[0]
    assert "beneficio" in unique_terms[1]
    assert "growth" in unique_terms[2]

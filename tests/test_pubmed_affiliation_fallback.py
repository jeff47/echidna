from xml.etree import ElementTree as ET

from app.ncbi import _find_authors, _normalize_affiliation_text


def test_single_article_affiliation_applied_to_all_missing_authors() -> None:
    article = ET.fromstring(
        """
        <PubmedArticle>
          <Article>
            <AuthorList>
              <Author>
                <LastName>Rice</LastName>
                <ForeName>Jeffrey</ForeName>
                <Initials>J</Initials>
              </Author>
              <Author>
                <LastName>Smith</LastName>
                <ForeName>Amy</ForeName>
                <Initials>A</Initials>
              </Author>
            </AuthorList>
            <Affiliation>Department of Immunology, Example Institute</Affiliation>
          </Article>
        </PubmedArticle>
        """
    )

    authors = _find_authors(article)

    assert len(authors) == 2
    assert authors[0].affiliation == "Department of Immunology, Example Institute"
    assert authors[1].affiliation == "Department of Immunology, Example Institute"


def test_multiple_article_affiliations_not_forced_onto_all_authors() -> None:
    article = ET.fromstring(
        """
        <PubmedArticle>
          <Article>
            <AuthorList>
              <Author>
                <LastName>Rice</LastName>
                <ForeName>Jeffrey</ForeName>
                <Initials>J</Initials>
              </Author>
              <Author>
                <LastName>Smith</LastName>
                <ForeName>Amy</ForeName>
                <Initials>A</Initials>
              </Author>
            </AuthorList>
            <Affiliation>Department A</Affiliation>
            <Affiliation>Department B</Affiliation>
          </Article>
        </PubmedArticle>
        """
    )

    authors = _find_authors(article)

    assert len(authors) == 2
    assert authors[0].affiliation == ""
    assert authors[1].affiliation == ""


def test_single_author_level_affiliation_propagates_to_missing_authors() -> None:
    article = ET.fromstring(
        """
        <PubmedArticle>
          <Article>
            <AuthorList>
              <Author>
                <LastName>Rice</LastName>
                <ForeName>Jeffrey</ForeName>
                <Initials>J</Initials>
                <AffiliationInfo>
                  <Affiliation>Division of Immunology, Example Institute</Affiliation>
                </AffiliationInfo>
              </Author>
              <Author>
                <LastName>Smith</LastName>
                <ForeName>Amy</ForeName>
                <Initials>A</Initials>
              </Author>
            </AuthorList>
          </Article>
        </PubmedArticle>
        """
    )

    authors = _find_authors(article)

    assert len(authors) == 2
    assert authors[0].affiliation == "Division of Immunology, Example Institute"
    assert authors[1].affiliation == "Division of Immunology, Example Institute"


def test_normalize_affiliation_text_adds_missing_spaces_for_numeric_markers() -> None:
    raw = "1Department of Microbiology and Immunology, and2 Department of Medicine, Albert Einstein College of Medicine, Bronx, New York, USA"
    normalized = _normalize_affiliation_text(raw)
    assert normalized.startswith("Department of Microbiology")
    assert "and Department of Medicine" not in normalized


def test_normalize_affiliation_text_drops_embedded_second_numbered_affiliation() -> None:
    raw = (
        "Departments of Medicine and Microbiology, Columbia University, New York, New York, USA. "
        "2 Department of Microbiology and Immunology, Albert Einstein College of Medicine, New York, New York, USA."
    )
    normalized = _normalize_affiliation_text(raw)
    assert "Columbia University" in normalized
    assert "Albert Einstein" not in normalized


def test_normalize_affiliation_text_collapses_merged_department_markers() -> None:
    raw = (
        "1Department of Microbiology and Immunology, and2 Department of Medicine, "
        "Albert Einstein College of Medicine, Bronx, New York, USA"
    )
    normalized = _normalize_affiliation_text(raw)
    assert normalized == (
        "Department of Microbiology and Immunology, "
        "Albert Einstein College of Medicine, Bronx, New York, USA"
    )


def test_normalize_affiliation_text_strips_grid_and_marker_prefixes() -> None:
    raw = (
        "grid.5386.8000000041936877 XWeill Cornell Medicine, New York, NY USA; "
        "grid.239915.50000 0001 2285 8823 Hospital for Special Surgery, E 70 th Street, New York, NY 10021 USA"
    )
    normalized = _normalize_affiliation_text(raw)
    assert "grid." not in normalized.lower()
    assert "XWeill" not in normalized
    assert "grid." not in normalized.lower()
    assert "0001 2285 8823" not in normalized
    assert "Weill Cornell Medicine" in normalized
    assert "E 70th Street" in normalized


def test_normalize_affiliation_text_strips_ror_url_and_id_fragments() -> None:
    raw = (
        "https://ror.org/03 zjqec80HSS Research Institute, Hospital for Special Surgery New York United States; "
        "https://ror.org/02 r109517Immunology and Microbial Pathogenesis Program, Weill Cornell Medicine New York United States"
    )
    normalized = _normalize_affiliation_text(raw)
    assert "https://ror.org" not in normalized
    assert "zjqec80" not in normalized
    assert "r109517" not in normalized
    assert "HSS Research Institute" in normalized
    assert "Immunology and Microbial Pathogenesis Program" in normalized


def test_normalize_affiliation_text_removes_single_letter_prefix_with_space() -> None:
    raw = "a Hospital for Special Surgery, Department of Rheumatology, E 70 th St, New York, NY 10021"
    normalized = _normalize_affiliation_text(raw)
    assert normalized.startswith("Hospital for Special Surgery")
    assert not normalized.startswith("a ")


def test_normalize_affiliation_text_strips_grid_artifacts_from_donlin_examples() -> None:
    raw = (
        "grid. 0001 2285 8823 Hospital for Special Surgery, E 70th Street, New York, NY 10009 USA; "
        "grid.5386. Weill Cornell Medicine, New York, NY USA"
    )
    normalized = _normalize_affiliation_text(raw)
    assert "grid" not in normalized.lower()
    assert normalized.startswith("Hospital for Special Surgery")
    assert "; Weill Cornell Medicine" in normalized


def test_normalize_affiliation_text_strips_superscript_marker_prefix() -> None:
    raw = "¹Department of Medicine, Example University, Boston, MA"
    normalized = _normalize_affiliation_text(raw)
    assert normalized.startswith("Department of Medicine")
    assert not normalized.startswith("1")


def test_normalize_affiliation_text_strips_punctuated_numeric_marker_prefix() -> None:
    raw = "1 . Department of Medicine, Example University, Boston, MA"
    normalized = _normalize_affiliation_text(raw)
    assert normalized.startswith("Department of Medicine")
    assert "1 ." not in normalized


def test_normalize_affiliation_text_preserves_address_numbers() -> None:
    raw = "Hospital for Special Surgery, E 70th Street, New York, NY 10021"
    normalized = _normalize_affiliation_text(raw)
    assert "70th Street" in normalized


def test_normalize_affiliation_text_strips_leading_numeric_marker_clusters() -> None:
    raw = "1 .38142. Transplantation Research Center, Renal Division, Brigham and Women's Hospital"
    normalized = _normalize_affiliation_text(raw)
    assert normalized.startswith("Transplantation Research Center")
    assert "1 .38142." not in normalized


def test_normalize_affiliation_text_strips_bracketed_numeric_markers() -> None:
    raw = "[2] Department of Medicine, Example University, Boston, MA"
    normalized = _normalize_affiliation_text(raw)
    assert normalized.startswith("Department of Medicine")
    assert "[2]" not in normalized

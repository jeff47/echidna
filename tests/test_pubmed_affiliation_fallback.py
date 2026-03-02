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

from xml.etree import ElementTree as ET

from app.ncbi import _find_authors


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

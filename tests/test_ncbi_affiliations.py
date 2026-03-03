from app.models import Author, Citation
from app.ncbi import NcbiClient, _extract_pmc_affiliation_maps, _extract_pmc_note_text_by_id, _extract_contrib_affiliation
from xml.etree import ElementTree as ET


def _author(position: int, last: str, fore: str, initials: str, affiliation: str) -> Author:
    return Author(position=position, last_name=last, fore_name=fore, initials=initials, affiliation=affiliation)




def test_extract_pmc_enrichment_parses_author_orcid() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <contrib-group>
            <contrib contrib-type="author">
              <name><surname>Rice</surname><given-names>Jeffrey</given-names></name>
              <contrib-id contrib-id-type="orcid">https://orcid.org/0000-0002-1825-0097</contrib-id>
              <xref ref-type="aff" rid="a1" />
            </contrib>
          </contrib-group>
          <aff id="a1">Department of Immunology, Institute A</aff>
        </article-meta>
      </front>
    </article>
    """
    client = NcbiClient()
    _, _, authors, _ = client._extract_pmc_enrichment(xml)

    assert len(authors) == 1
    assert authors[0].orcid == "0000-0002-1825-0097"

def test_extract_pmc_enrichment_includes_roles_and_affiliations() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <contrib-group>
            <contrib contrib-type="author">
              <name><surname>Rice</surname><given-names>Jeffrey</given-names></name>
              <xref ref-type="aff" rid="a1" />
              <xref ref-type="fn" rid="fn1" />
            </contrib>
            <contrib contrib-type="author">
              <name><surname>Smith</surname><given-names>Amy</given-names></name>
              <xref ref-type="aff" rid="a2" />
              <xref ref-type="fn" rid="fn2" />
            </contrib>
          </contrib-group>
          <aff id="a1">Department of Immunology, Institute A</aff>
          <aff id="a2">Department of Pathology, Institute B</aff>
          <author-notes>
            <fn id="fn1">These authors contributed equally to this work.</fn>
            <fn id="fn2">These authors are co-senior authors.</fn>
          </author-notes>
        </article-meta>
      </front>
    </article>
    """
    client = NcbiClient()
    co_first, co_senior, authors, pmc_count = client._extract_pmc_enrichment(xml)

    assert co_first == {1}
    assert co_senior == {2}
    assert authors[0].affiliation.startswith("Department of Immunology")
    assert authors[1].affiliation.startswith("Department of Pathology")
    assert pmc_count == 2
    assert authors[0].last_name == "Rice"
    assert authors[0].fore_name == "Jeffrey"


def test_extract_pmc_enrichment_detects_co_senior_from_corresp_note() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <contrib-group>
            <contrib contrib-type="author">
              <name><surname>Sokhi</surname><given-names>UK</given-names></name>
              <xref ref-type="fn" rid="eq1" />
            </contrib>
            <contrib contrib-type="author">
              <name><surname>Donlin</surname><given-names>Laura</given-names></name>
              <xref ref-type="corresp" rid="corr1" />
            </contrib>
            <contrib contrib-type="author">
              <name><surname>Ivashkiv</surname><given-names>Lionel</given-names></name>
              <xref ref-type="corresp" rid="corr1" />
            </contrib>
          </contrib-group>
          <author-notes>
            <fn id="eq1">These authors contributed equally.</fn>
            <corresp id="corr1">These authors share senior authorship.</corresp>
          </author-notes>
        </article-meta>
      </front>
    </article>
    """
    client = NcbiClient()
    co_first, co_senior, _, _ = client._extract_pmc_enrichment(xml)

    assert co_first == {1}
    assert co_senior == {2, 3}


def test_extract_pmc_enrichment_interprets_equal_prefix_as_co_first() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <contrib-group>
            <contrib contrib-type="author">
              <name><surname>A</surname><given-names>A1</given-names></name>
              <xref ref-type="fn" rid="eq1" />
            </contrib>
            <contrib contrib-type="author">
              <name><surname>B</surname><given-names>B1</given-names></name>
              <xref ref-type="fn" rid="eq1" />
            </contrib>
            <contrib contrib-type="author">
              <name><surname>C</surname><given-names>C1</given-names></name>
            </contrib>
            <contrib contrib-type="author">
              <name><surname>D</surname><given-names>D1</given-names></name>
            </contrib>
          </contrib-group>
          <author-notes>
            <fn id="eq1">These authors contributed equally.</fn>
          </author-notes>
        </article-meta>
      </front>
    </article>
    """
    client = NcbiClient()
    co_first, co_senior, _, _ = client._extract_pmc_enrichment(xml)

    assert co_first == {1, 2}
    assert co_senior == set()


def test_extract_pmc_enrichment_interprets_equal_suffix_as_co_senior() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <contrib-group>
            <contrib contrib-type="author">
              <name><surname>A</surname><given-names>A1</given-names></name>
            </contrib>
            <contrib contrib-type="author">
              <name><surname>B</surname><given-names>B1</given-names></name>
            </contrib>
            <contrib contrib-type="author">
              <name><surname>C</surname><given-names>C1</given-names></name>
              <xref ref-type="fn" rid="eq1" />
            </contrib>
            <contrib contrib-type="author">
              <name><surname>D</surname><given-names>D1</given-names></name>
              <xref ref-type="fn" rid="eq1" />
            </contrib>
          </contrib-group>
          <author-notes>
            <fn id="eq1">These authors contributed equally.</fn>
          </author-notes>
        </article-meta>
      </front>
    </article>
    """
    client = NcbiClient()
    co_first, co_senior, _, _ = client._extract_pmc_enrichment(xml)

    assert co_first == set()
    assert co_senior == {3, 4}


def test_extract_pmc_enrichment_detects_equal_contributions_phrase() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <contrib-group>
            <contrib contrib-type="author">
              <name><surname>A</surname><given-names>A1</given-names></name>
            </contrib>
            <contrib contrib-type="author">
              <name><surname>B</surname><given-names>B1</given-names></name>
            </contrib>
            <contrib contrib-type="author">
              <name><surname>C</surname><given-names>C1</given-names></name>
              <xref ref-type="fn" rid="eq2" />
            </contrib>
            <contrib contrib-type="author">
              <name><surname>D</surname><given-names>D1</given-names></name>
              <xref ref-type="fn" rid="eq2" />
            </contrib>
          </contrib-group>
          <author-notes>
            <fn id="eq2">The authors made equal contributions.</fn>
          </author-notes>
        </article-meta>
      </front>
    </article>
    """
    client = NcbiClient()
    co_first, co_senior, _, _ = client._extract_pmc_enrichment(xml)

    assert co_first == set()
    assert co_senior == {3, 4}


def test_extract_pmc_enrichment_uses_note_rid_even_with_nonstandard_ref_type() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <contrib-group>
            <contrib contrib-type="author">
              <name><surname>A</surname><given-names>A1</given-names></name>
            </contrib>
            <contrib contrib-type="author">
              <name><surname>B</surname><given-names>B1</given-names></name>
            </contrib>
            <contrib contrib-type="author">
              <name><surname>C</surname><given-names>C1</given-names></name>
              <xref ref-type="author-notes" rid="eq3" />
            </contrib>
            <contrib contrib-type="author">
              <name><surname>D</surname><given-names>D1</given-names></name>
              <xref ref-type="author-notes" rid="eq3" />
            </contrib>
          </contrib-group>
          <author-notes>
            <fn id="eq3">These authors contributed equally.</fn>
          </author-notes>
        </article-meta>
      </front>
    </article>
    """
    client = NcbiClient()
    co_first, co_senior, _, _ = client._extract_pmc_enrichment(xml)

    assert co_first == set()
    assert co_senior == {3, 4}


def test_fill_missing_affiliations_from_pmc_values() -> None:
    client = NcbiClient()
    citation = Citation(
        pmid="1",
        pmcid="PMC1",
        title="x",
        journal="y",
        print_year=2024,
        print_date="2024",
        authors=[
            _author(1, "Rice", "Jeffrey", "J", ""),
            _author(2, "Smith", "Amy", "A", "Already Present"),
        ],
        source_for_roles="pmc",
    )

    filled = client._fill_missing_affiliations(
        citation,
        {
            1: "Institute A",
            2: "Institute B",
        },
    )

    assert filled == 1
    assert citation.authors[0].affiliation == "Institute A"
    assert citation.authors[1].affiliation == "Already Present"


def test_replace_authors_from_pmc_prefers_pmc_author_list() -> None:
    client = NcbiClient()
    citation = Citation(
        pmid="2",
        pmcid="PMC2",
        title="x",
        journal="y",
        print_year=2024,
        print_date="2024",
        authors=[_author(1, "Rice", "Jeffrey", "J", "")],
        source_for_roles="pmc",
    )
    pmc_authors = [
        _author(1, "Rice", "Jeffrey", "J", "Institute A"),
        _author(2, "Smith", "Amy", "A", "Institute B"),
    ]

    changed = client._replace_authors_from_pmc(citation, pmc_authors)

    assert changed is True
    assert len(citation.authors) == 2
    assert citation.authors[0].affiliation == "Institute A"


def test_extract_contrib_affiliation_handles_multi_rid_and_labels() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <contrib-group>
            <contrib contrib-type="author">
              <name><surname>Rice</surname><given-names>Jeffrey</given-names></name>
              <xref ref-type="aff" rid="a1 a2">1,2</xref>
            </contrib>
          </contrib-group>
          <aff id="a1"><label>1</label>Department of Immunology, Institute A</aff>
          <aff id="a2"><label>2</label>Department of Medicine, Institute B</aff>
        </article-meta>
      </front>
    </article>
    """
    root = ET.fromstring(xml)
    contrib = root.find(".//contrib")
    assert contrib is not None
    aff_by_id, aff_by_label = _extract_pmc_affiliation_maps(root)
    note_by_id = _extract_pmc_note_text_by_id(root)

    affiliation = _extract_contrib_affiliation(
        contrib,
        aff_by_id=aff_by_id,
        aff_by_label=aff_by_label,
        note_by_id=note_by_id,
    )

    assert "Department of Immunology" in affiliation
    assert "Department of Medicine" in affiliation


def test_extract_contrib_affiliation_prefers_rid_over_label_text() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <contrib-group>
            <contrib contrib-type="author">
              <name><surname>Rice</surname><given-names>Jeffrey</given-names></name>
              <xref ref-type="aff" rid="a1">1,2</xref>
            </contrib>
          </contrib-group>
          <aff id="a1"><label>1</label>Department of Medicine, Columbia University, New York, USA.</aff>
          <aff id="a2"><label>2</label>Department of Microbiology, Albert Einstein College of Medicine, Bronx, NY, USA.</aff>
        </article-meta>
      </front>
    </article>
    """
    root = ET.fromstring(xml)
    contrib = root.find(".//contrib")
    assert contrib is not None
    aff_by_id, aff_by_label = _extract_pmc_affiliation_maps(root)
    note_by_id = _extract_pmc_note_text_by_id(root)

    affiliation = _extract_contrib_affiliation(
        contrib,
        aff_by_id=aff_by_id,
        aff_by_label=aff_by_label,
        note_by_id=note_by_id,
    )

    assert "Columbia University" in affiliation
    assert "Albert Einstein" not in affiliation


def test_extract_contrib_affiliation_uses_fn_note_when_affiliation_like() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <contrib-group>
            <contrib contrib-type="author">
              <name><surname>Rice</surname><given-names>Jeffrey</given-names></name>
              <xref ref-type="fn" rid="fn_aff" />
            </contrib>
          </contrib-group>
          <author-notes>
            <fn id="fn_aff">Division of Allergy and Immunology, NIH, Bethesda, MD, USA.</fn>
          </author-notes>
        </article-meta>
      </front>
    </article>
    """
    root = ET.fromstring(xml)
    contrib = root.find(".//contrib")
    assert contrib is not None
    aff_by_id, aff_by_label = _extract_pmc_affiliation_maps(root)
    note_by_id = _extract_pmc_note_text_by_id(root)

    affiliation = _extract_contrib_affiliation(
        contrib,
        aff_by_id=aff_by_id,
        aff_by_label=aff_by_label,
        note_by_id=note_by_id,
    )

    assert "NIH" in affiliation


def test_extract_contrib_affiliation_prefers_direct_aff_over_note_fallback() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <contrib-group>
            <contrib contrib-type="author">
              <name><surname>Rice</surname><given-names>Jeffrey</given-names></name>
              <xref ref-type="aff" rid="a1" />
              <xref ref-type="fn" rid="fn_aff" />
            </contrib>
          </contrib-group>
          <aff id="a1">Department of Medicine, Columbia University, New York, USA.</aff>
          <author-notes>
            <fn id="fn_aff">Department of Microbiology, Albert Einstein College of Medicine, Bronx, NY, USA.</fn>
          </author-notes>
        </article-meta>
      </front>
    </article>
    """
    root = ET.fromstring(xml)
    contrib = root.find(".//contrib")
    assert contrib is not None
    aff_by_id, aff_by_label = _extract_pmc_affiliation_maps(root)
    note_by_id = _extract_pmc_note_text_by_id(root)

    affiliation = _extract_contrib_affiliation(
        contrib,
        aff_by_id=aff_by_id,
        aff_by_label=aff_by_label,
        note_by_id=note_by_id,
    )

    assert "Columbia University" in affiliation
    assert "Albert Einstein" not in affiliation


def test_extract_pmc_affiliation_maps_prefers_article_meta_aff_when_ids_repeat() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <aff id="a1">Department of Medicine, Columbia University, New York, USA.</aff>
        </article-meta>
      </front>
      <body>
        <sec>
          <aff id="a1">Department of Microbiology, Albert Einstein College of Medicine, Bronx, NY, USA.</aff>
        </sec>
      </body>
    </article>
    """
    root = ET.fromstring(xml)
    article_meta = root.find(".//front/article-meta")
    assert article_meta is not None

    aff_by_id, _ = _extract_pmc_affiliation_maps(article_meta)

    assert aff_by_id["a1"].startswith("Department of Medicine, Columbia University")

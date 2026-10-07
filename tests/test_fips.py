"""Offline tests for county name -> FIPS. A tiny fixture stands in for the
Census file, using the real rows whose spellings trip up naive matching."""

import pytest

from src.fips import FipsLookup, county_fips, normalize_county, state_postal

FIXTURE = """STATE|STATEFP|COUNTYFP|COUNTYNS|COUNTYNAME|CLASSFP|FUNCSTAT
MO|29|189|00758549|St. Louis County|H1|A
MO|29|510|00767557|St. Louis city|C7|F
MO|29|186|00758547|Ste. Genevieve County|H1|A
MO|29|095|00758502|Jackson County|H1|A
IL|17|031|01784766|Cook County|H1|A
IL|17|099|00835871|LaSalle County|H1|A
LA|22|033|00558530|East Baton Rouge Parish|H1|A
AK|02|110|01419973|Juneau City and Borough|H6|A
NM|35|013|00929105|Doña Ana County|H1|A
VA|51|019|01480101|Bedford County|H1|A
VA|51|750|01498434|Radford city|C7|A
KS|20|209|00485065|Wyandotte County|H6|A
"""


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "national_county2020.txt"
    path.write_text(FIXTURE, encoding="utf-8")
    return path


@pytest.fixture
def lookup(reference):
    return FipsLookup.from_file(reference, download=False)


@pytest.mark.parametrize("name", ["Saint Louis", "St. Louis", "st louis county",
                                  "ST. LOUIS COUNTY", "St Louis"])
def test_st_louis_county_spellings(lookup, name):
    assert lookup.county_fips("MO", name) == "29189"


@pytest.mark.parametrize("name", ["St. Louis city", "Saint Louis City", "st louis CITY"])
def test_independent_city_is_not_the_county(lookup, name):
    assert lookup.county_fips("Missouri", name) == "29510"


def test_state_by_name_or_postal_any_case():
    assert state_postal("missouri") == "MO"
    assert state_postal("mo") == "MO"
    assert state_postal("District of Columbia") == "DC"
    with pytest.raises(KeyError):
        state_postal("Narnia")


@pytest.mark.parametrize("state,name,fips", [
    ("IL", "Cook", "17031"),
    ("IL", "La Salle", "17099"),                 # LaSalle vs La Salle
    ("LA", "East Baton Rouge", "22033"),         # Parish suffix optional
    ("AK", "Juneau", "02110"),                   # City and Borough
    ("NM", "Dona Ana", "35013"),                 # accent-insensitive
    ("MO", "Sainte Genevieve", "29186"),         # Sainte -> Ste.
    ("VA", "Bedford City", "51019"),             # merged city -> county
    ("VA", "Radford", "51750"),                  # bare name of a city
])
def test_suffixes_accents_and_fallbacks(lookup, state, name, fips):
    assert lookup.county_fips(state, name) == fips


def test_unknown_county_raises_instead_of_guessing(lookup):
    with pytest.raises(KeyError):
        lookup.county_fips("MO", "Cook")         # Cook is in IL, not MO


def test_states_with_supports_cross_state_sewersheds(lookup):
    assert lookup.states_with("Wyandotte") == ["KS"]
    assert lookup.states_with("Nowhere") == []


def test_normalize_is_stable():
    assert normalize_county("St. Louis County") == normalize_county("Saint Louis") == "stlouis"


def test_module_function_uses_cached_reference(reference):
    assert county_fips("Illinois", "Cook County", reference=reference) == "17031"

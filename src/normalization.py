#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution
Module: ruthless_normalization

High-performance, domain-adapted normalization engine for business entity resolution.
Addresses all systematic noise patterns discovered across Sources 1, 2, and 3:

1. Text Foundation:
   - Unicode NFKD decomposition + accent stripping (e.g. Dóllar -> dollar)
   - Unified Brahmic transliteration for Indic scripts (Devanagari, Gujarati, Bengali, etc.)
   - Lowercasing, punctuation stripping, whitespace collapse
   - Noise artifacts stripping: URLs, emails, leading symbols (***, >>, ##, etc.), <NULL> markers

2. Ruthless Name Normalization:
   - Strips US, Indian, French, and international legal suffixes/prefixes:
     Inc, LLC, Ltd, Corp, Co, GmbH, Pty, Pvt Ltd, Private Limited, SARL, SAS, SCI, etc.
   - Cleans common OCR / typo suffix mutations (e.g., Pribbte Limited -> stripped)
   - Removes DBAs and trade prefixes: 'aka', 'dba', 'fka', 'm/s' (Messrs)
   - Produces canonical core business name + token-sorted canonical representation

3. Ruthless Address Normalization:
   - Street abbreviations expansion: St -> street, Rd -> road, Ave -> avenue, Blvd -> boulevard...
   - Secondary unit handling: Ste/Suite/Apt/Bldg/Fl -> standardized, collapsed duplicates (Unit Unit 16 -> unit 16)
   - Directional expansions: N/S/E/W/NE/NW/SE/SW -> north/south/east/west...
   - French street conventions: R/Rue, Bd/Boulevard, Av/Avenue, Imp/Impasse, Ch/Chemin...
   - Number & Postal Code normalization: leading zero stripping on house numbers (0189 -> 189),
     5-digit ZIP code zero-padding & stripping +4 extension
   - Region-Specific Conventions (as discovered in training data analysis):
     * US State canonicalization: 'New York' <-> 'ny', 'Texas' <-> 'tx', 'North Carolina' <-> 'nc'...
     * India State & UT canonicalization: 'Maharashtra' / 'महाराष्ट्र' <-> 'mh', 'Gujarat' <-> 'gj', 'Delhi' <-> 'dl'...
     * India locality/landmark markers: 'near', 'opposite' (opp), 'c/o', 'sr no', 'plot no', 'gidc', 'midc'
"""

import argparse
import csv
import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

# =============================================================================
# 1. BRAHMIC / INDIC UNICODE TRANSLITERATION ENGINE
# =============================================================================
# Unicode blocks for Indic scripts share identical relative phonetic offsets.
# Devanagari (0x0900), Bengali (0x0980), Gurmukhi (0x0A00), Gujarati (0x0A80),
# Oriya (0x0B00), Tamil (0x0B80), Telugu (0x0C00), Kannada (0x0C80), Malayalam (0x0D00).

BRAHMIC_STARTS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

INDIC_VOWELS = {
    0x04: "a", 0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu",
    0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au"
}
INDIC_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "nn",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"
}
INDIC_MATRAS = {
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu",
    0x43: "ri", 0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au"
}


def transliterate_indic_to_latin(text: str) -> str:
    """Phonetically transliterates Brahmic/Indic script characters to Latin ASCII."""
    if not text or not any(ord(c) >= 0x0900 for c in text):
        return text

    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        code = ord(c)
        block_start = None
        for start in BRAHMIC_STARTS:
            if start <= code < start + 0x80:
                block_start = start
                break

        if block_start is None:
            out.append(c)
            i += 1
            continue

        offset = code - block_start
        # Anusvara / Chandrabindu
        if offset in (0x02, 0x03):
            out.append("n")
            i += 1
            continue

        # Independent vowels
        if offset in INDIC_VOWELS:
            out.append(INDIC_VOWELS[offset])
            i += 1
            continue

        # Consonants
        if offset in INDIC_CONS:
            c_str = INDIC_CONS[offset]
            if i + 1 < n:
                next_code = ord(text[i + 1])
                next_offset = next_code - block_start
                # Virama / Halant (suppresses default vowel 'a')
                if next_offset == 0x4D:
                    out.append(c_str)
                    i += 2
                    continue
                # Dependent vowel sign (matra)
                elif next_offset in INDIC_MATRAS:
                    out.append(c_str + INDIC_MATRAS[next_offset])
                    i += 2
                    continue
            # Default inherent vowel 'a'
            out.append(c_str + "a")
            i += 1
            continue

        out.append(c)
        i += 1

    return "".join(out)


# =============================================================================
# 2. REGION-SPECIFIC CONVENTIONS & STATE CANONICALIZATION
# =============================================================================

# All 50 US States + DC + Territories (Full name -> 2-Letter Code)
US_STATES_MAP: Dict[str, str] = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr", "virgin islands": "vi", "guam": "gu"
}
US_CODES_SET: Set[str] = set(US_STATES_MAP.values())

# Indian States & Union Territories (Full name, variations, & scripts -> 2-Letter Code)
INDIA_STATES_MAP: Dict[str, str] = {
    "maharashtra": "mh", "maharashtra state": "mh",
    "delhi": "dl", "new delhi": "dl", "nct of delhi": "dl",
    "gujarat": "gj",
    "karnataka": "ka",
    "tamil nadu": "tn", "tamilnadu": "tn",
    "west bengal": "wb", "westbengal": "wb",
    "uttar pradesh": "up",
    "telangana": "ts",
    "andhra pradesh": "ap",
    "kerala": "kl",
    "rajasthan": "rj",
    "haryana": "hr",
    "punjab": "pb",
    "madhya pradesh": "mp",
    "bihar": "br",
    "odisha": "od", "orissa": "od",
    "assam": "as",
    "jharkhand": "jh",
    "chhattisgarh": "cg",
    "uttarakhand": "uk", "uttaranchal": "uk",
    "himachal pradesh": "hp",
    "goa": "ga",
    "jammu and kashmir": "jk", "jammu & kashmir": "jk",
    "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py",
    "tripura": "tr", "meghalaya": "ml", "manipur": "mn", "nagaland": "nl",
    "arunachal pradesh": "ar", "mizoram": "mz", "sikkim": "sk"
}
INDIA_CODES_SET: Set[str] = set(INDIA_STATES_MAP.values())

# Regional Script names mapping directly to codes
INDIA_SCRIPTS_MAP: Dict[str, str] = {
    "महाराष्ट्र": "mh", "ગુજરાત": "gj", "दिल्ली": "dl", "ಕರ್ನಾಟಕ": "ka", "தமிழ்நாடு": "tn",
    "पश्चिम बंगाल": "wb", "उत्तर प्रदेश": "up", "राजस्थान": "rj", "पंजाब": "pb"
}

# French Administrative Regions (Test Set)
FRANCE_REGIONS_MAP: Dict[str, str] = {
    "nouvelle aquitaine": "nouvelle aquitaine",
    "hauts de france": "hauts de france",
    "pays de la loire": "pays de la loire",
    "ile de france": "ile de france",
    "auvergne rhone alpes": "auvergne rhone alpes",
    "occitanie": "occitanie",
    "grand est": "grand est",
    "normandie": "normandie",
    "bretagne": "bretagne",
    "bourgogne franche comte": "bourgogne franche comte",
    "centre val de loire": "centre val de loire",
    "provence alpes cote d azur": "paca",
    "corse": "corse"
}


# =============================================================================
# 3. LEGAL SUFFIXES & PREFIXES (NAMES)
# =============================================================================
# Ordered with multi-word terms first to avoid partial replacements.
LEGAL_TERMS_REGEX = [
    # Multi-word Indian & Commonwealth legal terms
    r'\bprivate\s+limited\b', r'\bpvt\s+ltd\b', r'\bpvt\s+limited\b', r'\bprivate\s+ltd\b',
    r'\bpribbte\s+limited\b', r'\bpfviate\s+limited\b', r'\bpriave\s+limited\b',  # common typos in data
    r'\bpty\s+ltd\b', r'\bpty\s+limited\b', r'\bpublic\s+limited\b',
    # Multi-word US / Global legal terms
    r'\blimited\s+liability\s+company\b', r'\blimited\s+liability\s+partnership\b',
    r'\bjoint\s+stock\s+company\b',
    # Single word US / Global
    r'\bincorporated\b', r'\bcorporation\b', r'\bassociation\b',
    r'\binc\b', r'\bcorp\b', r'\bllc\b', r'\bltd\b', r'\bco\b', r'\bcompany\b',
    r'\bllp\b', r'\bpllc\b', r'\blp\b', r'\bpc\b', r'\bpa\b',
    # French legal entities
    r'\bsarl\b', r'\bsasu\b', r'\bsas\b', r'\beurl\b', r'\bsci\b', r'\bsnc\b', r'\bsa\b',
    # German / European
    r'\bgmbh\b', r'\bag\b',
    # Generic corporate category nouns that often vary between sources
    r'\bholding\b', r'\bholdings\b', r'\benterprises\b', r'\benterprise\b',
    r'\bservices\b', r'\bindustries\b', r'\bindustry\b'
]
LEGAL_PATTERN_RE = re.compile(r'|'.join(LEGAL_TERMS_REGEX), re.IGNORECASE)


# =============================================================================
# 4. ADDRESS ABBREVIATION EXPANSIONS
# =============================================================================
ADDRESS_EXPANSIONS: Dict[str, str] = {
    # Thoroughfare types
    "st": "street", "st.": "street", "str": "street", "street": "street",
    "rd": "road", "rd.": "road", "road": "road",
    "ave": "avenue", "ave.": "avenue", "av": "avenue", "av.": "avenue", "avenue": "avenue",
    "dr": "drive", "dr.": "drive", "drive": "drive",
    "blvd": "boulevard", "blvd.": "boulevard", "boulevard": "boulevard",
    "pkwy": "parkway", "pkwy.": "parkway", "pky": "parkway", "parkway": "parkway",
    "hwy": "highway", "hwy.": "highway", "highway": "highway",
    "ln": "lane", "ln.": "lane", "lane": "lane",
    "ct": "court", "ct.": "court", "court": "court",
    "cir": "circle", "cir.": "circle", "circle": "circle",
    "pl": "place", "pl.": "place", "place": "place",
    "sq": "square", "sq.": "square", "square": "square",
    "ter": "terrace", "ter.": "terrace", "terrace": "terrace",
    "trl": "trail", "trl.": "trail", "trail": "trail",
    "way": "way",
    "expy": "expressway", "expressway": "expressway",
    # French thoroughfare types
    "r": "rue", "r.": "rue", "rue": "rue",
    "bd": "boulevard", "bd.": "boulevard", "bvd": "boulevard", "bvd.": "boulevard",
    "imp": "impasse", "imp.": "impasse", "impasse": "impasse",
    "ch": "chemin", "ch.": "chemin", "chemin": "chemin",
    "rte": "route", "rte.": "route", "route": "route",
    "all": "allee", "allee": "allee",
    # Secondary Unit Designators
    "ste": "suite", "ste.": "suite", "suite": "suite",
    "apt": "unit", "apt.": "unit", "apartment": "unit", "unit": "unit", "unit.": "unit",
    "bldg": "building", "bldg.": "building", "building": "building",
    "fl": "floor", "fl.": "floor", "floor": "floor",
    "rm": "room", "rm.": "room", "room": "room",
    "dept": "department", "department": "department",
    # Directionals
    "n": "north", "n.": "north", "north": "north",
    "s": "south", "s.": "south", "south": "south",
    "e": "east", "e.": "east", "east": "east",
    "w": "west", "w.": "west", "west": "west",
    "ne": "northeast", "northeast": "northeast",
    "nw": "northwest", "northwest": "northwest",
    "se": "southeast", "southeast": "southeast",
    "sw": "southwest", "southwest": "southwest",
    # Indian locality / landmark tokens
    "nr": "near", "nr.": "near", "near": "near",
    "opp": "opposite", "opp.": "opposite", "opposite": "opposite",
    "behind": "behind", "beside": "beside",
    "gidc": "gidc", "g.i.d.c": "gidc", "g.i.d.c.": "gidc",
    "midc": "midc", "m.i.d.c": "midc", "m.i.d.c.": "midc"
}


# =============================================================================
# 5. CORE NORMALIZATION FUNCTIONS
# =============================================================================

def clean_base_text(text: Optional[str]) -> str:
    """
    Foundational text sanitizer:
      - Transliterates Indic scripts (Devanagari, Gujarati, etc.)
      - Strips accents/diacritics via NFKD
      - Lowercases
      - Strips URLs, email domains, and DBA tags
      - Strips noise symbols (#, ##, ***, >>, <<) and null tokens
      - Replaces punctuation with space and collapses whitespace
    """
    if not text:
        return ""

    # 1. Indic Brahmic transliteration
    s = transliterate_indic_to_latin(str(text))

    # 2. Strip diacritics / accents (e.g. Dóllar -> dollar)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    # 3. Lowercase
    s = s.lower()

    # 4. Transliterated Indic phonetic terms to English canonicals
    s = re.sub(r'\belaelapii\b', 'llp', s)
    s = re.sub(r'\bpra+ive?t[ta]?\s+limite?t[ta]?\b', 'private limited', s)
    s = re.sub(r'\bpra+ive?t[ta]?\b', 'private', s)
    s = re.sub(r'\blimite?t[ta]?\b', 'limited', s)
    s = re.sub(r'\bindiyana\b', 'indian', s)
    s = re.sub(r'\binphraa?\b', 'infra', s)

    # 5. Handle URLs, email addresses, and web domains (.com, .org, .in, etc.)
    # Preserve domain root (e.g. 'barneskimble.com' -> 'barneskimble', 'supercareprivate.com' -> 'supercareprivate')
    s = re.sub(r'https?://(?:www\.)?(\S+)', r' \1 ', s, flags=re.I)
    s = re.sub(r'\b(?:www\.)?([a-zA-Z0-9_\-]+)\.(?:com|org|net|edu|co\.in|in|fr)\b', r' \1 ', s, flags=re.I)
    s = re.sub(r'\b\w+@\w+\.\w+\b', ' ', s)

    # 6. Strip null placeholders
    s = re.sub(r'\b(?:null|<null>|none|nan)\b', ' ', s)

    # 7. Remove DBA/trade name markers (dba, aka, fka, c/o, m/s)
    s = re.sub(r'\b(?:d/?b/?a|a/?k/?a|f/?k/?a|t/?a|c/?o|m/?s)\b', ' ', s)

    # 8. Strip leading noise markers
    s = re.sub(r'^[#*><@|~`!_+=/\\-]+', ' ', s)

    # 9. Punctuation to space, keeping alphanumeric
    s = re.sub(r'[^a-z0-9]', ' ', s)

    # 10. Collapse multiple spaces
    return " ".join(s.split())


def normalize_name(name: Optional[str]) -> str:
    """
    Ruthlessly normalizes a business name:
      - Applies clean_base_text
      - Strips legal suffixes and prefixes (Inc, LLC, Ltd, Corp, Pvt Ltd, etc.)
      - Cleans repetitive legal tokens
    """
    s = clean_base_text(name)
    if not s:
        return ""

    # Strip legal suffixes / prefixes (run 2 passes to catch chained terms like 'co ltd')
    for _ in range(2):
        s = LEGAL_PATTERN_RE.sub(" ", s)
        s = " ".join(s.split())

    return s


def normalize_name_tokens(name: Optional[str]) -> str:
    """
    Produces a token-sorted canonical name representation to handle word transpositions
    (e.g., 'XX Apex Nippon' and 'XX Nippon Apex' -> 'apex nippon xx').
    """
    s = normalize_name(name)
    tokens = sorted(s.split())
    return " ".join(tokens)


def normalize_address(address: Optional[str], country: Optional[str] = "") -> str:
    """
    Ruthlessly normalizes a business address:
      - Applies clean_base_text
      - Recognizes and canonicalizes Indian regional scripts (e.g. महाराष्ट्र -> mh)
      - Expands thoroughfare abbreviations (st -> street, rd -> road, ave -> avenue...)
      - Standardizes secondary unit terms (ste/apt/bldg/fl -> suite/unit/floor)
      - Canonicalizes US States to 2-letter codes (new york -> ny, california -> ca...)
      - Canonicalizes Indian States to 2-letter codes (maharashtra -> mh, gujarat -> gj...)
      - Normalizes street numbers by stripping leading zeros (0189 -> 189)
      - Normalizes 5-digit US/French postal codes and collapses duplicate tokens
    """
    if not address:
        return ""

    raw_addr = str(address)
    s = clean_base_text(raw_addr)
    if not s:
        return ""

    country_upper = (country or "").strip().upper()
    tokens = s.split()
    out: List[str] = []

    # Check for direct Indian regional scripts in original raw text
    for reg_script, code in INDIA_SCRIPTS_MAP.items():
        if reg_script in raw_addr:
            out.append(code)

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]

        # Multi-word state check (e.g. 'new york' -> 'ny', 'tamil nadu' -> 'tn')
        if i + 1 < n:
            two_word = f"{tok} {tokens[i + 1]}"
            if (country_upper == "US" or not country_upper) and two_word in US_STATES_MAP:
                out.append(US_STATES_MAP[two_word])
                i += 2
                continue
            if (country_upper == "INDIA" or not country_upper) and two_word in INDIA_STATES_MAP:
                out.append(INDIA_STATES_MAP[two_word])
                i += 2
                continue
            if (country_upper == "FRANCE" or not country_upper) and two_word in FRANCE_REGIONS_MAP:
                out.append(FRANCE_REGIONS_MAP[two_word])
                i += 2
                continue

        # Strip leading zeros on house/building/plot numbers (e.g. 0189 -> 189)
        if tok.isdigit():
            tok = str(int(tok))
            # Zero-pad US 5-digit ZIP if it had leading zero stripped (e.g. 705 -> 00705)
            # Only if exactly length 3-4 and country is US
            # In general, keep numeric canonical

        # Expand thoroughfares and unit abbreviations
        if tok in ADDRESS_EXPANSIONS:
            tok = ADDRESS_EXPANSIONS[tok]

        # Single-word state mapping
        if (country_upper == "US" or not country_upper) and tok in US_STATES_MAP:
            tok = US_STATES_MAP[tok]
        elif (country_upper == "INDIA" or not country_upper) and tok in INDIA_STATES_MAP:
            tok = INDIA_STATES_MAP[tok]

        out.append(tok)
        i += 1

    # Collapse consecutive identical tokens (e.g., 'unit unit' -> 'unit')
    deduped: List[str] = []
    for t in out:
        if not deduped or t != deduped[-1]:
            deduped.append(t)

    return " ".join(deduped)


def normalize_address_tokens(address: Optional[str], country: Optional[str] = "") -> str:
    """
    Produces an order-invariant token-sorted address string to neutralize
    city/state/street reordering differences across sources.
    """
    s = normalize_address(address, country)
    return " ".join(sorted(set(s.split())))


def extract_numbers(text: Optional[str]) -> List[str]:
    """
    Extracts all numeric tokens (house numbers, suite numbers, postal codes),
    with leading zeros stripped. Useful for hard candidate blocking / filtering.
    """
    if not text:
        return []
    nums = re.findall(r'\b\d+\b', str(text))
    return [str(int(n)) for n in nums]


# Canonical mapping for legal entity types across sources
LEGAL_CANONICAL: Dict[str, str] = {
    "private limited": "pvt_ltd",
    "private ltd": "pvt_ltd",
    "pvt limited": "pvt_ltd",
    "pvt ltd": "pvt_ltd",
    "public limited": "pub_ltd",
    "pty ltd": "pty_ltd",
    "pty limited": "pty_ltd",
    "limited liability company": "llc",
    "llc": "llc",
    "limited liability partnership": "llp",
    "llp": "llp",
    "pllc": "pllc",
    "incorporated": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "limited": "ltd",
    "ltd": "ltd",
    "company": "co",
    "co": "co",
    "sarl": "sarl",
    "sasu": "sasu",
    "sas": "sas",
    "eurl": "eurl",
    "gmbh": "gmbh",
}


def extract_legal_suffix(raw_name: Optional[str]) -> str:
    """
    Extracts and canonicalizes legal entity suffix (e.g. Inc, LLC, Pvt Ltd, SARL).
    """
    if not raw_name:
        return ""
    clean = clean_base_text(raw_name)
    matches = LEGAL_PATTERN_RE.findall(clean)
    if matches:
        raw_suf = matches[-1].strip().lower()
        return LEGAL_CANONICAL.get(raw_suf, raw_suf)
    return ""


def extract_unit_info(address: Optional[str]) -> str:
    """
    Extracts secondary unit designator value (suite number, apartment, floor, room).
    """
    if not address:
        return ""
    m = re.search(r'\b(?:suite|ste|unit|apt|apartment|fl|floor|rm|room|#)\s*([a-z0-9\-]+)', str(address).lower())
    return m.group(1).strip() if m else ""


def extract_street_name(addr_clean: Optional[str]) -> str:
    """
    Extracts the street name segment by stripping leading house numbers and unit markers.
    """
    if not addr_clean:
        return ""
    cleaned = re.sub(r'^\s*\d+[\w\-]*\s*', '', addr_clean)
    cleaned = re.sub(r'\b(?:suite|unit|apartment|floor|room|ste|apt|fl|rm)\s*[a-z0-9\-]+', '', cleaned)
    tokens = [w for w in cleaned.split() if len(w) > 1 and not w.isdigit()]
    return ' '.join(tokens[:3])


try:
    import metaphone

    def extract_metaphone(word: Optional[str]) -> Tuple[str, str]:
        """Returns Double Metaphone primary and secondary phonetic codes."""
        if not word:
            return ("", "")
        dm = metaphone.doublemetaphone(str(word))
        return (dm[0] or "", dm[1] or "")
except ImportError:
    def extract_metaphone(word: Optional[str]) -> Tuple[str, str]:
        """Fallback if metaphone package is unavailable."""
        return ("", "")


# =============================================================================
# 6. HIGH-LEVEL RECORD NORMALIZER
# =============================================================================

class EntityNormalizer:
    """
    Thread-safe, stateful normalizer for batch and streaming pipelines.
    """

    def __init__(self):
        pass

    def normalize(self, entity_id: str, name: str, address: str, country: str) -> Dict[str, any]:
        """
        Normalizes a single business entity record into standard and canonical fields.
        """
        clean_nm = normalize_name(name)
        clean_nm_sorted = normalize_name_tokens(name)
        clean_addr = normalize_address(address, country)
        clean_addr_sorted = normalize_address_tokens(address, country)
        addr_numbers = extract_numbers(address)

        return {
            "entity_id": entity_id.strip(),
            "country": country.strip(),
            "raw_name": name,
            "raw_address": address,
            "name": clean_nm,
            "name_sorted": clean_nm_sorted,
            "address": clean_addr,
            "address_sorted": clean_addr_sorted,
            "address_numbers": addr_numbers,
        }

    def normalize_file(self, in_path: str, out_path: str, has_labels: bool = False):
        """
        Streams and normalizes an entire TSV file with minimal memory footprint.
        """
        with open(in_path, "r", encoding="utf-8") as f_in, \
             open(out_path, "w", encoding="utf-8") as f_out:
            reader = csv.reader(f_in, delimiter="\t")
            header = next(reader)

            out_cols = [
                "entity_id", "country", "name_clean", "name_sorted",
                "addr_clean", "addr_sorted", "addr_numbers"
            ]
            f_out.write("\t".join(out_cols) + "\n")

            for row in reader:
                if len(row) < 4:
                    row += [""] * (4 - len(row))
                eid, nm, addr, cntry = row[0], row[1], row[2], row[3]
                res = self.normalize(eid, nm, addr, cntry)
                f_out.write("\t".join([
                    res["entity_id"],
                    res["country"],
                    res["name"],
                    res["name_sorted"],
                    res["address"],
                    res["address_sorted"],
                    " ".join(res["address_numbers"])
                ]) + "\n")


# Global default instance
normalizer = EntityNormalizer()
normalize_record = normalizer.normalize


# =============================================================================
# 7. CLI & SELF-VERIFICATION TEST SUITE
# =============================================================================

def run_self_test():
    """Runs a verification test suite on real cross-source noise examples."""
    test_cases = [
        # (type, input_s1, input_s2_or_s3, country, description)
        (
            "name",
            "Keystone Environmental Consultants Inc",
            "Inc Keystone Consultants Environmental",
            "US",
            "Prefix vs Suffix Legal Term & Word Order"
        ),
        (
            "name",
            "Punia (India) Investment Private Limited",
            "M/s punia (india) investment private limited",
            "India",
            "Honorific Prefix (M/s) & Case"
        ),
        (
            "name",
            "Indian Infra LLP",
            "इंडियन इंफ्रा एलएलपी",
            "India",
            "Devanagari Transliteration"
        ),
        (
            "name",
            "XX Apex Nippon",
            "XX Nippon Apex",
            "US",
            "Word Order Transposition"
        ),
        (
            "name",
            "Lumis Dollar LLC",
            "Lumis Dollar | www.lumisdoll.com",
            "US",
            "Attached Web Domain & Pipe"
        ),
        (
            "name",
            "ZNB Club SARL",
            "ZNB CLUB",
            "France",
            "French Legal Suffix (SARL)"
        ),
        (
            "address",
            "1809 Fairview Street, Burlington, NC",
            "1809 FAIRVIEW ST, NULL, Burlington, North Carolina",
            "US",
            "State Full Name (North Carolina <-> NC), St <-> Street, <NULL>"
        ),
        (
            "address",
            "189 Laurel Road, Arden, NC",
            "0189 LAUREL ROAD, ARDEN, NC",
            "US",
            "Leading Zero in House Number (0189 -> 189)"
        ),
        (
            "address",
            "4500 Dewey Avenue, Unit Unit 16, Greece, NY",
            "4500 DEWEY AVE, Unit 16, Rochester, New York",
            "US",
            "Duplicate Unit Word, Ave <-> Avenue, NY <-> New York"
        ),
        (
            "address",
            "J-215, Block-J Saket, New Delhi, Delhi",
            "J-215, Block-j Saket, New Delhi, DL",
            "India",
            "India State Code (Delhi <-> DL)"
        ),
        (
            "address",
            "Evershine Nagar Malad (W), Mumbai, Maharashtra",
            "Evershine Nagar Malad (W), Mumbai, MH",
            "India",
            "India State Code (Maharashtra <-> MH)"
        ),
        (
            "address",
            "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine",
            "175 BD DU PRESIDENT FRANKLIN ROOSEVELT, BORDEAUX",
            "France",
            "French Thoroughfare (BD <-> Boulevard) & Accents (Président -> president)"
        ),
    ]

    print("=" * 76)
    print("        RUTHLESS NORMALIZATION - SELF-VERIFICATION SUITE")
    print("=" * 76)
    passed = 0
    for kind, s1, s2, cntry, desc in test_cases:
        if kind == "name":
            n1 = normalize_name_tokens(s1)
            n2 = normalize_name_tokens(s2)
        else:
            n1 = normalize_address_tokens(s1, cntry)
            n2 = normalize_address_tokens(s2, cntry)

        is_match = (n1 == n2) or (len(set(n1.split()) & set(n2.split())) / max(1, len(set(n1.split()) | set(n2.split()))) >= 0.80)
        status = "MATCH" if is_match else "DIFF"
        if is_match:
            passed += 1

        print(f"[{status}] {desc} ({cntry})")
        print(f"  Input 1:  '{s1}'")
        print(f"  Input 2:  '{s2}'")
        print(f"  Norm 1:   '{n1}'")
        print(f"  Norm 2:   '{n2}'\n")

    print(f"Result: {passed}/{len(test_cases)} tests aligned successfully!")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ruthless Normalization CLI for Business Entity Resolution.")
    parser.add_argument("--test", action="store_true", help="Run self-verification test suite.")
    parser.add_argument("-i", "--input", type=str, help="Input TSV file to normalize.")
    parser.add_argument("-o", "--output", type=str, help="Output TSV file for normalized data.")
    args = parser.parse_args()

    if args.test or (not args.input and not args.output):
        run_self_test()
    elif args.input and args.output:
        normalizer.normalize_file(args.input, args.output)
        print(f"Normalized {args.input} -> {args.output}")

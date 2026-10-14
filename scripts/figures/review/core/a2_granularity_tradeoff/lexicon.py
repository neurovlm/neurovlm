"""Lexicon detector of study-specific elements in a generated map description: details a map alone cannot reveal, so
the model must have taken them from its (title, abstract)-style training texts.

Each category is one alternation of the patterns below (case-insensitive unless prefixed "CS:"). A text's count per
category is its number of non-overlapping matches, so a span counts once per category (it can count in two categories,
e.g. "our study"). ``count(text)``, ``matches(text)``, ``count_frame(texts)`` and ``top_terms(texts)``.
"""
from __future__ import annotations

import re
from typing import Iterable

_NUM_WORDS = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
              r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)(?:[- ](?:one|two|three|four|five|six|"
              r"seven|eight|nine))?")
_PEOPLE = (r"(?:participants?|subjects?|volunteers?|individuals?|patients?|adults?|children|adolescents?|controls?|men|women|males?|"
           r"females?|people|persons|infants?|students?|listeners?|readers?|monkeys?|rats?|mice|twins?|dyads?|smokers?|users?|athletes?|"
           r"musicians?|experts?|novices?|cases|veterans?|survivors?|siblings?|participants)")
_MOD = r"(?:(?:healthy|right-handed|left-handed|young|younger|older|elderly|adult|male|female|native|typically[- ]developing|normal|unmedicated|medicated|drug-naive|first-episode|chronic|human|neurotypical)\s+)*"

LEXICON: dict[str, list[str]] = {
    "sample_size": [
        r"\b[nN]\s*=\s*\d+",
        rf"\b\d{{1,5}}\s+{_MOD}{_PEOPLE}\b",
        rf"\b{_NUM_WORDS}\s+{_MOD}{_PEOPLE}\b",
        r"\b(?:sample|cohort|group)\s+of\s+(?:\d+|" + _NUM_WORDS + r")\b",
    ],
    "age": [
        r"\baged?\s+(?:between\s+)?\d+", r"\bmean\s+age\b", r"\bages?\s+(?:range|ranged|between|from)\b", r"\bage\s*[=:]\s*\d+",
        r"\b\d+\s*(?:-|–|to)\s*\d+\s*(?:-\s*)?(?:years?|yrs?)\b", r"\b\d+\s*(?:-\s*)?(?:years?|yrs?)[- ]old\b", r"\b\d+\s+years\s+of\s+age\b",
        r"\b(?:children|adolescents?|infants?|toddlers?|neonates?|newborns?|preterm|preschool(?:ers)?|school-aged|older\s+adults|"
        r"elderly|young\s+adults|middle-aged|teenagers?|youths?)\b",
    ],
    "clinical": [
        r"\bpatients?\b", r"\bschizophreni\w*", r"\bschizoaffective\b", r"\bpsychos[ie]s\b", r"\bpsychotic\b",
        r"\bmajor\s+depressi\w*", r"\bdepressive\s+disorder\b", r"\bdepressed\b", "CS:" r"\b(?:MDD|ASD|ADHD|PTSD|OCD|MCI|TBI|AD|PD|MS|HIV|BPD)\b",
        r"\bautis\w*", r"\battention[- ]deficit\b", r"\bhyperactivity\s+disorder\b", r"\bparkinson\w*", r"\balzheimer\w*", r"\bdementia\b",
        r"\bmild\s+cognitive\s+impairment\b", r"\bepilep\w*", r"\bseizures?\b", r"\bbipolar\b", r"\bpost-?traumatic\b", r"\bobsessive",
        r"\banxiety\s+disorders?\b", r"\bsocial\s+anxiety\b", r"\bpanic\s+disorder\b", r"\bphobi\w*", r"\bstrokes?\b",
        r"\bmultiple\s+sclerosis\b", r"\bhuntington\w*", r"\btourette\w*", r"\bdyslexi\w*", r"\baphasi\w*", r"\banorexi\w*", r"\bbulimi\w*",
        r"\beating\s+disorders?\b", r"\baddict\w*", r"\bsubstance\s+(?:use|abuse|dependence)\b", r"\balcohol(?:ism|ic|ics)?\b",
        r"\bcocaine\b", r"\bheroin\b", r"\bopioids?\b", r"\bcannabis\b", r"\bnicotine\s+dependen\w*", r"\bchronic\s+pain\b",
        r"\bfibromyalgia\b", r"\bmigraine\w*", r"\btinnitus\b", r"\btraumatic\s+brain\s+injur\w*", r"\bconcussions?\b", r"\bcoma\w*",
        r"\bvegetative\b", r"\bamnes\w*", r"\bhemispatial\s+neglect\b", r"\blesion\s+patients\b", r"\bglioma\w*", r"\btumou?rs?\b",
        r"\bcancer\b", r"\bdiabet\w*", r"\bobes\w*", r"\binsomnia\b", r"\bborderline\s+personality\b", r"\bpsychopath\w*",
        r"\bhealthy\s+controls?\b", r"\bclinical\s+(?:populations?|samples?|groups?|trials?)\b", r"\bdisorders?\b", r"\bdiseases?\b",
        r"\bsyndromes?\b", r"\bdiagnos\w*",
    ],
    "task": [
        r"\bstroop\b", r"\b(?:\d|one|two|three|n)[- ]back\b", r"\bgo/?\s?no-?go\b", r"\bgo-nogo\b", r"\bstop[- ]signal\b", r"\bflanker\b",
        r"\boddball\b", r"\bsternberg\b", r"\bposner\b", r"\bsimon\s+task\b", r"\bwisconsin\s+card\b", "CS:" r"\bWCST\b",
        r"\btower\s+of\s+(?:london|hanoi)\b", r"\biowa\s+gambling\b", r"\bballoon\s+analog(?:ue)?\s+risk\b", "CS:" r"\bBART\b",
        r"\bmonetary\s+incentive\s+delay\b", "CS:" r"\bMID\s+task\b", r"\bdelay\s+discounting\b", r"\bultimatum\s+game\b", r"\bdictator\s+game\b",
        r"\bprisoner'?s\s+dilemma\b", r"\btrust\s+game\b", r"\bmental\s+rotation\s+task\b", r"\bdigit\s+span\b",
        r"\bserial\s+reaction\s+time\b", r"\bfinger[- ]tapping\s+task\b", r"\banti-?saccade\s+task\b", r"\bvisual\s+search\s+task\b",
        r"\battentional\s+blink\b", r"\bdot[- ]probe\b", r"\bface[- ]matching\s+task\b", r"\bfalse[- ]belief\s+task\b",
        r"\breading\s+the\s+mind\s+in\s+the\s+eyes\b", r"\bimplicit\s+association\s+test\b", r"\blexical\s+decision\b",
        r"\bsemantic\s+decision\s+task\b", r"\bverb\s+generation\b", r"\bpicture\s+naming\b", r"\bcued\s+recall\b",
        r"\bpaired[- ]associates?\b", r"\b(?:the|a|an)\s+[\w-]+\s+(?:task|paradigm)\s+(?:was|were|in\s+which|that\s+required)\b",
    ],
    "stimulus": [
        r"\b\d+(?:\.\d+)?\s*(?:ms|msec|milliseconds?)\b", r"\b\d+(?:\.\d+)?\s+s\b(?![-\w])", r"\b\d+(?:\.\d+)?\s*(?:sec|secs|seconds?)\b", r"\b\d+(?:\.\d+)?\s*k?Hz\b",
        r"\b\d+\s+(?:trials|blocks|runs|sessions|stimuli|images|pictures|photographs|words|sentences|faces|items|events|videos|clips|tones)\b",
        "CS:" r"\b(?:IAPS|NimStim|KDEF|Ekman)\b", r"\binternational\s+affective\s+picture\s+system\b", r"\b(?:was|were)\s+(?:presented|shown|exposed)\b",
        r"\bblock(?:ed)?[- ]design\b", r"\bevent-related\s+design\b", r"\b(?:video|movie|film)\s+clips?\b",
    ],
    "scanner": [
        "CS:" r"(?<![\w.])\d(?:\.\d)?\s*-?\s*T(?![\w-])", r"\bTesla\b", "CS:" r"\bTR\b", "CS:" r"\bTE\b", r"\brepetition\s+time\b", r"\becho\s+time\b",
        r"\bSiemens\b", r"\bPhilips\b", r"\bGeneral\s+Electric\b", "CS:" r"\bGE\s+(?:Signa|scanner|Healthcare|Discovery)\b", r"\bSigna\b",
        r"\b(?:Trio|Prisma|Skyra|Verio|Achieva|Ingenia|Allegra|Magnetom)\b", r"\becho[- ]planar\b", "CS:" r"\bEPI\b", r"\bmulti-?band\b",
        r"\bflip\s+angle\b", r"\bvoxel\s+size\b", r"\b\d+(?:\.\d+)?\s*mm\s*(?:isotropic|thick|slices?|cubic)\b",
        r"\b\d+\s+(?:axial\s+|contiguous\s+|oblique\s+|interleaved\s+)?slices\b", r"\bhead\s+coil\b", r"\b\d+-channel\b",
        r"\bfield\s+of\s+view\b", "CS:" r"\bFOV\b",
    ],
    "statistics": [
        r"\bp\s*[<>=≤≥]\s*0?\.\d+", r"\bp\s*[<>=]\s*\d(?:\.\d+)?\s*[x×]\s*10", r"\bp-?values?\b", "CS:" r"\b(?:FDR|FWE|FWER)\b", r"\bBonferroni\b",
        r"\bfamily-?wise\b", r"\bfalse\s+discovery\b", r"\bcluster[- ](?:level|corrected|wise|extent|threshold|size|based|forming)\b",
        r"\b(?:un)?corrected\b", "CS:" r"\b[tzF]\s*\(\s*\d+(?:\s*,\s*\d+)?\s*\)\s*=", "CS:" r"\b[zZtT]\s*[=>]\s*-?\d+(?:\.\d+)?\b", r"\br\s*=\s*-?0?\.\d+",
        r"\bcohen'?s\s+d\b", r"\beffect\s+sizes?\b", r"\bodds\s+ratios?\b", r"\bconfidence\s+intervals?\b", r"\bANOVAs?\b", r"\bt-tests?\b",
        r"\bstatistically\b", r"\b(?:reached|did\s+not\s+reach|failed\s+to\s+reach)\s+significance\b",
        # statistical "significant(ly)" only (not "emotionally / behaviourally significant events", "a significant role")
        r"\bsignificant(?:ly)?\s+(?:greater|higher|lower|more|less|larger|smaller|increased|decreased|reduced|enhanced|elevated|stronger|"
        r"weaker|activation|activations|activity|increases?|decreases?|differences?|different|correlat\w*|associat\w*|effects?|"
        r"reductions?|changes?|interactions?|main\s+effects?|group|clusters?|positive|negative|relationships?|predictors?)\b",
        r"\b(?:showed|show|shows|revealed|found|observed|was|were)\s+(?:a\s+|no\s+)?significant(?:ly)?\b",
    ],
    "first_person": [
        # generic "how we perceive ..." / "our environment" is NOT counted: only "we" + a study verb and "our" + a study noun
        r"\bwe\s+(?:also\s+|further\s+|then\s+|first\s+|here\s+)?(?:found|find|show|showed|shown|demonstrate|demonstrated|report|reported|examined|"
        r"examine|investigated|investigate|used|use|measured|tested|test|scanned|recruited|observed|identified|compared|propose|proposed|"
        r"hypothesi[sz]ed|predicted|assessed|conducted|performed|analy[sz]ed|aimed|sought|explored|studied|applied|employed|present|"
        r"presented|provide|describe|developed|acquired|collected|recorded|asked|trained|manipulated|characterized|reveal|revealed)\b",
        r"\bour\s+(?:study|results|findings|data|sample|participants|subjects|patients|analysis|analyses|experiment|experiments|work|"
        r"research|approach|method|methods|hypothesis|hypotheses|paradigm|cohort|observations|investigation)\b",
        r"\b(?:participants|subjects|patients|volunteers|individuals|children|adults)\s+(?:were|was|had|performed|completed|underwent|viewed|"
        r"watched|listened|rated|received|showed|exhibited|made|responded)\b",
        r"\bwere\s+(?:recruited|scanned|asked|instructed|randomi[sz]ed|randomly\s+assigned|enrolled|tested|trained|imaged|included|excluded)\b",
    ],
    "this_study": [
        r"\b(?:this|the\s+present|the\s+current|our|the\s+present-day)\s+(?:study|studies|paper|work|experiment|experiments|investigation|"
        r"research|article|report|analysis|trial|sample|data|dataset)\b",
        r"\bhere,?\s+we\b", r"\bin\s+this\s+(?:study|paper|work|article|report|experiment)\b",
        r"\b(?:the|these|our)\s+(?:results|findings|data)\s+(?:show|showed|suggest|suggested|indicate|indicated|reveal|revealed|demonstrate|"
        r"demonstrated|support|supported|provide|provided)\b",
        r"\bthe\s+(?:study|experiment|authors)\s+(?:shows?|showed|found|finds|demonstrates?|reveals?|examined|investigated|tested|aimed)\b",
    ],
}
CATEGORIES: tuple[str, ...] = tuple(LEXICON)
_COMPILED: dict[str, re.Pattern] = {}


def _compile(cat: str) -> re.Pattern:
    if cat not in _COMPILED:
        parts = [f"(?:{p[3:]})" if p.startswith("CS:") else f"(?i:{p})" for p in LEXICON[cat]]
        _COMPILED[cat] = re.compile("|".join(parts))
    return _COMPILED[cat]


def matches(text: str) -> dict[str, list[str]]:
    t = str(text)
    return {c: [m.group(0) for m in _compile(c).finditer(t)] for c in CATEGORIES}


def count(text: str) -> dict[str, int]:
    t = str(text)
    return {c: sum(1 for _ in _compile(c).finditer(t)) for c in CATEGORIES}


def count_frame(texts: Iterable[str]):
    """Per-text counts (one column per category) plus ``total`` and ``any`` (>= 1 element)."""
    import pandas as pd

    df = pd.DataFrame([count(t) for t in texts], columns=list(CATEGORIES))
    df["total"] = df[list(CATEGORIES)].sum(axis=1)
    df["any"] = df["total"] > 0
    return df


def top_terms(texts: Iterable[str], k: int = 8) -> dict[str, list[tuple[str, int]]]:
    """Most frequent matched strings per category (lower-cased)."""
    from collections import Counter

    cnt = {c: Counter() for c in CATEGORIES}
    for t in texts:
        for c, ms in matches(t).items():
            cnt[c].update(m.lower().strip() for m in ms)
    return {c: cnt[c].most_common(k) for c in CATEGORIES}

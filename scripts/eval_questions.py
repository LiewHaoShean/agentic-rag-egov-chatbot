"""Fixed evaluation question set for the offline RAG evaluation harness.

Thirty queries spanning the four ingested agencies and three of the four
supported languages, so retrieval routing can be measured per agency AND per
language. Kept in a separate module so the question set is version-controlled
and the same fixed set is reused across runs (results are only comparable if
the questions do not drift).

`expected_category` is the agency whose documents SHOULD be retrieved. It is
the ground truth for the automated routing metric; the 2/1/0 relevance rubric
is scored manually against the retrieved text.
"""
from __future__ import annotations

QUESTIONS: list[dict[str, str]] = [
    # ------------------------------------------------------------------ KWSP
    {"id": "Q01", "lang": "en", "category": "epf",
     "q": "What is the EPF contribution rate for employees?"},
    {"id": "Q02", "lang": "en", "category": "epf",
     "q": "Can I withdraw my EPF savings at age 55?"},
    {"id": "Q03", "lang": "ms", "category": "epf",
     "q": "Apakah itu Akaun Persaraan?"},
    {"id": "Q04", "lang": "zh", "category": "epf",
     "q": "什么是公积金？"},
    {"id": "Q05", "lang": "en", "category": "epf",
     "q": "How do I nominate a beneficiary for my EPF savings?"},
    {"id": "Q06", "lang": "ms", "category": "epf",
     "q": "Bagaimana cara menyemak baki simpanan KWSP saya?"},
    {"id": "Q07", "lang": "en", "category": "epf",
     "q": "What is i-Saraan and who is eligible to join?"},
    {"id": "Q08", "lang": "en", "category": "epf",
     "q": "What happens to my EPF contributions if I keep working after 55?"},

    # ------------------------------------------------------------------ LHDN
    {"id": "Q09", "lang": "en", "category": "tax",
     "q": "Who is required to file Borang BE?"},
    {"id": "Q10", "lang": "en", "category": "tax",
     "q": "What is the deadline for submitting an individual income tax return?"},
    {"id": "Q11", "lang": "ms", "category": "tax",
     "q": "Bagaimana cara mendaftar nombor cukai pendapatan?"},
    {"id": "Q12", "lang": "en", "category": "tax",
     "q": "What tax relief can I claim for medical expenses?"},
    {"id": "Q13", "lang": "en", "category": "tax",
     "q": "What is PCB and how is it deducted from my salary?"},
    {"id": "Q14", "lang": "en", "category": "tax",
     "q": "How do I apply for a certificate of residence?"},
    {"id": "Q15", "lang": "ms", "category": "tax",
     "q": "Apakah e-Filing dan bagaimana saya menggunakannya?"},
    {"id": "Q16", "lang": "en", "category": "tax",
     "q": "What is the corporate income tax rate in Malaysia?"},

    # --------------------------------------------------------------- PERKESO
    {"id": "Q17", "lang": "en", "category": "socso",
     "q": "How do I register as an employer with PERKESO?"},
    {"id": "Q18", "lang": "en", "category": "socso",
     "q": "What is covered under the employment injury scheme?"},
    {"id": "Q19", "lang": "ms", "category": "socso",
     "q": "Bolehkah pekerja sendiri menyertai PERKESO?"},
    {"id": "Q20", "lang": "en", "category": "socso",
     "q": "How do I make a claim for an employment injury?"},
    {"id": "Q21", "lang": "en", "category": "socso",
     "q": "What is the invalidity pension scheme?"},
    {"id": "Q22", "lang": "ms", "category": "socso",
     "q": "Berapakah kadar caruman PERKESO untuk majikan?"},
    {"id": "Q23", "lang": "en", "category": "socso",
     "q": "What happens if an employer fails to register employees with PERKESO?"},

    # ------------------------------------------------------------------- JPN
    {"id": "Q24", "lang": "en", "category": "identity",
     "q": "How do I replace a lost MyKad?"},
    {"id": "Q25", "lang": "en", "category": "identity",
     "q": "How much does it cost to replace a lost identity card?"},
    {"id": "Q26", "lang": "ms", "category": "identity",
     "q": "Bagaimana cara mendaftarkan kelahiran anak?"},
    {"id": "Q27", "lang": "en", "category": "identity",
     "q": "What documents are needed to register a marriage?"},
    {"id": "Q28", "lang": "en", "category": "identity",
     "q": "How do I register a death in Malaysia?"},
    {"id": "Q29", "lang": "ms", "category": "identity",
     "q": "Apakah syarat untuk memohon kewarganegaraan Malaysia?"},
    {"id": "Q30", "lang": "zh", "category": "identity",
     "q": "如何更换遗失的身份证？"},
]

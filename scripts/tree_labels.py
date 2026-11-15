"""German labels for the tree diagrams.

Sources: UCI datasets 144 and 350; Home Credit columns description.
"""

FEATURES = {
    "Kontostatus": "Girokontostatus", "Laufzeit_Monate": "Laufzeit (Monate)",
    "Kredithistorie": "Kredithistorie", "Verwendungszweck": "Verwendungszweck",
    "Kreditbetrag_DM": "Kreditbetrag (DM)", "Sparguthaben": "Sparguthaben",
    "Beschaeftigungsdauer": "Beschäftigungsdauer", "Ratenanteil_Prozent": "Ratenanteil (%)",
    "Familienstand_Geschlecht": "Familienstand / Geschlecht",
    "Weitere_Schuldner_Buergen": "Weitere Schuldner / Bürgen", "Wohnsitzdauer": "Wohnsitzdauer",
    "Vermoegen": "Vermögen", "Alter_Jahre": "Alter (Jahre)",
    "Weitere_Ratenplaene": "Weitere Ratenpläne", "Wohnsituation": "Wohnsituation",
    "Anzahl_Kredite_bei_Bank": "Kredite bei dieser Bank", "Beruf": "Beruf",
    "Anzahl_Unterhaltsberechtigte": "Unterhaltsberechtigte", "Telefon": "Telefon",
    "Auslaendischer_Arbeitnehmer": "Ausländischer Arbeitnehmer",
    "LIMIT_BAL": "Kreditrahmen (NT-Dollar)", "AGE": "Alter (Jahre)",
    "SEX": "Geschlecht (Code)", "EDUCATION": "Bildungsabschluss (Code)",
    "MARRIAGE": "Familienstand (Code)",
    "EXT_SOURCE_1": "Externer Score 1", "EXT_SOURCE_2": "Externer Score 2",
    "EXT_SOURCE_3": "Externer Score 3", "AMT_INCOME_TOTAL": "Gesamteinkommen",
    "AMT_CREDIT": "Kreditbetrag", "AMT_ANNUITY": "Kreditannuität",
    "AMT_GOODS_PRICE": "Warenpreis", "CODE_GENDER": "Geschlecht",
    "REGION_RATING_CLIENT_W_CITY": "Regionsbewertung inkl. Stadt (Code)",
}
for suffix, month in [("0", "September"), ("2", "August"), ("3", "Juli"),
                      ("4", "Juni"), ("5", "Mai"), ("6", "April")]:
    FEATURES[f"PAY_{suffix}"] = f"Zahlungsstatus {month} (Code)"
for number, month in enumerate(["September", "August", "Juli", "Juni", "Mai", "April"], 1):
    FEATURES[f"BILL_AMT{number}"] = f"Rechnungsbetrag {month} (NT-Dollar)"
    FEATURES[f"PAY_AMT{number}"] = f"Zahlungsbetrag {month} (NT-Dollar)"

CATEGORIES = {
    "A11": "Girokontostand unter 0 DM", "A12": "Girokontostand 0 bis unter 200 DM",
    "A13": "Mindestens 200 DM / Gehaltsabtretung", "A14": "Kein Girokonto vorhanden",
    "A30": "Keine Kredite / alle ordnungsgemäß zurückgezahlt",
    "A31": "Alle Kredite bei dieser Bank ordnungsgemäß zurückgezahlt",
    "A32": "Bestehende Kredite bisher ordnungsgemäß bedient",
    "A33": "Frühere Zahlungsverzögerungen",
    "A34": "Kritische Kredithistorie / weitere Kredite bei anderen Banken",
    "A40": "Neuwagen", "A41": "Gebrauchtwagen", "A42": "Möbel / Ausstattung",
    "A43": "Radio / Fernsehen", "A44": "Haushaltsgeräte", "A45": "Reparaturen",
    "A46": "Ausbildung", "A48": "Umschulung", "A49": "Geschäftlich", "A410": "Sonstiges",
    "A61": "Unter 100 DM", "A62": "100 bis unter 500 DM", "A63": "500 bis unter 1.000 DM",
    "A64": "Mindestens 1.000 DM", "A65": "Unbekannt / kein Sparkonto",
    "A71": "Arbeitslos", "A72": "Unter 1 Jahr", "A73": "1 bis unter 4 Jahre",
    "A74": "4 bis unter 7 Jahre", "A75": "Mindestens 7 Jahre",
    "A101": "Keine weiteren Schuldner / Bürgen", "A102": "Mitantragsteller", "A103": "Bürge",
    "A141": "Bank", "A142": "Geschäft", "A143": "Keine weiteren Ratenpläne",
    "A151": "Miete", "A152": "Eigentum", "A153": "Kostenlos",
    "A191": "Kein Telefon", "A192": "Telefon auf eigenen Namen",
    "A201": "Ja", "A202": "Nein",
    "missing": "Fehlender Wert", "M": "Männlich", "F": "Weiblich",
}

TARGETS = {
    "light_cardclient": {1: "gutes Kreditrisiko", 2: "schlechtes Kreditrisiko"},
    "mid_cardclients": {0: "kein Zahlungsausfall", 1: "Zahlungsausfall"},
    "hard_cardclient": {0: "keine Zahlungsschwierigkeiten", 1: "Zahlungsschwierigkeiten"},
}


def question(feature, threshold, categorical_columns):
    """Return the question and whether "Ja" is the left child.

    One-hot splits are asked as "category present?", so "Ja" is the right child there.
    """
    feature = feature.split("__", 1)[-1]
    for column in sorted(categorical_columns, key=len, reverse=True):
        if feature.startswith(column + "_") and 0 <= threshold < 1:
            category = feature[len(column) + 1:]
            meaning = CATEGORIES.get(category, category)
            if category in {"A14", "A33", "A34", "A143"}:
                return meaning + "?", False
            label = FEATURES.get(column, column.replace("_", " "))
            return f"{label}: {meaning}?", False
    label = FEATURES.get(feature, feature.replace("_", " "))
    number = f"{threshold:.6g}".replace(".", ",")
    return f"{label} ≤ {number}?", True

import io
import re
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import unicodedata

st.set_page_config(
    page_title="Haushaltsbuch Dashboard",
    page_icon="💶",
    layout="wide",
)

# ------------------------------
# Helpers
# ------------------------------
def _norm_amount(series: pd.Series) -> pd.Series:
    def parse_one(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        s = re.sub(r"[^\d,\.\-\+ ]", "", s)
        s = s.replace(" ", "").replace(".", "")
        if "," in s and s.count(",") == 1:
            s = s.replace(",", ".")
        if s.endswith("-") and not s.startswith("-"):
            s = "-" + s[:-1]
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.apply(parse_one)


def _normalize_header(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = (s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss"))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {_normalize_header(c): c for c in df.columns}
    for cand in candidates:
        cn = _normalize_header(cand)
        for nkey, orig in norm_map.items():
            if cn == nkey or cn in nkey:
                return orig
    return None


def normalize_schema(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.empty:
        return df

    date_col = _pick_first(df, ["buchung","buchungsdatum","datum","date","wertstellungsdatum","wertstellung","valuta"])
    book_col = _pick_first(df, ["wertstellungsdatum","wertstellung","valuta"])
    amt_col  = _pick_first(df, ["betrag","umsatzbetrag","amount"])
    cur_col  = _pick_first(df, ["währung","waehrung","currency","waehrungscode","eur"])
    usage1   = _pick_first(df, ["verwendungszweck","verwendung","verwendungszweck1","verwendungszweck2"])
    text_col = _pick_first(df, ["buchungstext","text","vermerk"])
    cp_col   = _pick_first(df, ["auftraggeber","auftraggeber/empfänger","auftraggeberempfaenger","zahlungsempfänger","zahlungseingang","beguenstigter","empfänger","gegenkonto","counterparty"])
    iban_col = _pick_first(df, ["iban","kontonummer","account"])

    out = pd.DataFrame()
    if date_col is not None:
        out["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    else:
        out["date"] = pd.NaT

    out["booking_date"] = pd.to_datetime(df[book_col], errors="coerce", dayfirst=True, infer_datetime_format=True) if book_col else pd.NaT

    if amt_col is not None:
        out["amount"] = _norm_amount(df[amt_col])
    else:
        out["amount"] = np.nan

    if cur_col is not None:
        cur_series = df[cur_col].astype(str).str.strip().replace({"nan": ""})
        cur_series = cur_series.where(cur_series!="", other="EUR")
        out["currency"] = cur_series
    else:
        out["currency"] = "EUR"

    pieces = []
    if text_col: pieces.append(df[text_col].astype(str))
    if usage1:   pieces.append(df[usage1].astype(str))
    desc = None
    if pieces:
        desc = pieces[0]
        for p in pieces[1:]:
            desc = desc.fillna("") + " | " + p.fillna("")
    out["description"] = desc.astype(str) if desc is not None else ""

    if cp_col:
        out["counterparty"] = df[cp_col].astype(str)
    else:
        out["counterparty"] = ""

    account = df[iban_col].astype(str) if iban_col else source_name
    out["account"] = account

    out["purpose"] = df[usage1].astype(str) if usage1 else ""

    out = out.dropna(subset=["amount"])
    out["amount"] = out["amount"].astype(float)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["year"] = out["date"].dt.year
    out["flow"] = np.where(out["amount"] < 0, "expense", "income")

    for col in ["description", "counterparty", "account", "purpose", "currency"]:
        out[col] = out[col].astype(str).str.strip()
    return out


def load_rules_from_csv(file) -> pd.DataFrame:
    try:
        rules = pd.read_csv(file)
        if not {"pattern", "category"}.issubset(set(rules.columns)):
            st.warning("Regel-CSV braucht Spalten: pattern, category")
            return pd.DataFrame(columns=["pattern", "category"])
        return rules.dropna(subset=["pattern", "category"])
    except Exception as e:
        st.warning(f"Konnte Regeln nicht laden: {e}")
        return pd.DataFrame(columns=["pattern", "category"])


def apply_categories(df: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = "Unkategorisiert"
    if rules is None or rules.empty:
        return df
    hay = (
        df["counterparty"].fillna("").astype(str) + " | " +
        df["description"].fillna("").astype(str) + " | " +
        df["purpose"].fillna("").astype(str)
    ).str.lower()

    for _, row in rules.iterrows():
        patt = str(row["pattern"])
        cat  = str(row["category"])
        try:
            mask = hay.str.contains(patt, case=False, regex=True, na=False)
        except Exception:
            mask = False
        df.loc[mask, "category"] = cat
    return df


def aggregate(df: pd.DataFrame):
    by_month_cat = (
        df.groupby(["month", "category"])["amount"]
        .sum().reset_index()
        .sort_values(["month", "amount"])
    )
    expenses = df[df["amount"] < 0]
    by_cat_total = expenses.groupby("category")["amount"].sum().sort_values().reset_index()
    by_account = df.groupby("account")["amount"].sum().sort_values().reset_index()

    df["sign"] = np.where(df["amount"] < 0, "Ausgaben", "Einnahmen")
    in_out_month = df.groupby(["month", "sign"])["amount"].sum().reset_index()

    return by_month_cat, by_cat_total, by_account, in_out_month


# ------------------------------
# UI
# ------------------------------
st.title("💶 Haushaltsbuch Dashboard")
st.caption("Lade deine Bank-CSV/XLSX-Dateien hoch, wende optionale Kategorienregeln an und erhalte mobilfreundliche Auswertungen.")

uploaded = st.file_uploader("Bank-Exporte (mehrere Dateien erlaubt)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
rules_file = st.file_uploader("Optional: Regeln (CSV mit Spalten pattern, category)", type=["csv"])

if uploaded:
    st.info("Nach dem Upload siehst du unten eine **Header‑Mapping‑Preview** pro Datei.")
    frames = []
    for up in uploaded:
        try:
            name_lower = up.name.lower()
            if name_lower.endswith((".xlsx", ".xls")):
                # Excel: erstes Blatt
                df = pd.read_excel(up)  # engines auto
            else:
                content = up.read()
                df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")

            # Mapping-Preview
            low = {c.lower(): c for c in df.columns}
            def _pick(cands):
                from collections import OrderedDict
                for c in cands:
                    for l, o in low.items():
                        if c in l:
                            return o
                return None
            preview = {
                "date": _pick(["buchung","buchungsdatum","datum","date","wertstellungsdatum","wertstellung","valuta"]),
                "booking_date": _pick(["wertstellungsdatum","wertstellung","valuta"]),
                "amount": _pick(["betrag","umsatzbetrag","amount"]),
                "currency": _pick(["währung","waehrung","currency","waehrungscode","eur"]),
                "description_part1": _pick(["buchungstext","text","vermerk"]),
                "description_part2": _pick(["verwendungszweck","verwendung"]),
                "counterparty": _pick(["auftraggeber","auftraggeber/empfänger","auftraggeberempfaenger","zahlungsempfänger","zahlungseingang","beguenstigter","empfänger","gegenkonto","counterparty"]),
                "account": _pick(["iban","kontonummer","account"]),
            }
            st.markdown(f"**{up.name}** – erkannte Spalten:")
            st.json(preview)

            norm = normalize_schema(df, source_name=up.name)
            norm["source_file"] = up.name
            frames.append(norm)
        except Exception as e:
            st.error(f"Fehler beim Einlesen von {up.name}: {e}")

    if frames:
        raw = pd.concat(frames, ignore_index=True).sort_values("date")

        # Zeitraumfilter optional
        st.subheader("Filter")
        use_date_filter = st.checkbox("Zeitraum filtern", value=False)
        if use_date_filter and not raw.empty:
            min_d, max_d = raw["date"].min().date(), raw["date"].max().date()
            start, end = st.date_input("Zeitraum", value=(min_d, max_d))
            if isinstance(start, date) and isinstance(end, date):
                mask = (raw["date"].dt.date >= start) & (raw["date"].dt.date <= end)
                raw = raw[mask]

        # Regeln anwenden
        rules_df = pd.DataFrame(columns=["pattern","category"])
        if rules_file:
            try:
                rules_df = pd.read_csv(rules_file)
            except Exception as e:
                st.warning(f"Regeln konnten nicht gelesen werden: {e}")
        cat_df = apply_categories(raw, rules_df)

        st.subheader("Rohdaten")
        st.dataframe(cat_df, use_container_width=True)

        by_month_cat, by_cat_total, by_account, in_out_month = aggregate(cat_df)

        st.subheader("Auswertung")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Summe je Monat**")
            month_totals = by_month_cat.groupby("month")["amount"].sum().reset_index().sort_values("month")
            st.dataframe(month_totals, use_container_width=True)

            fig1, ax1 = plt.subplots()
            ax1.plot(month_totals["month"], month_totals["amount"])
            ax1.set_title("Summe (Einnahmen/Ausgaben) je Monat")
            ax1.set_xlabel("Monat")
            ax1.set_ylabel("Betrag")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig1, use_container_width=True)

        with c2:
            st.markdown("**Top 10 Ausgabenkategorien**")
            top10 = by_cat_total.head(10)
            st.dataframe(top10, use_container_width=True)
            fig2, ax2 = plt.subplots()
            ax2.barh(top10["category"], top10["amount"])
            ax2.set_title("Top 10 Ausgabenkategorien (negativ)")
            ax2.set_xlabel("Betrag")
            ax2.set_ylabel("Kategorie")
            st.pyplot(fig2, use_container_width=True)

        st.markdown("**Pro Konto (Summe)**")
        st.dataframe(by_account, use_container_width=True)

        st.markdown("**Einnahmen vs. Ausgaben je Monat**")
        fig3, ax3 = plt.subplots()
        for sign, grp in in_out_month.groupby("sign"):
            ax3.plot(grp["month"], grp["amount"], label=sign)
        ax3.set_title("Einnahmen vs. Ausgaben je Monat")
        ax3.set_xlabel("Monat")
        ax3.set_ylabel("Betrag")
        ax3.legend()
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig3, use_container_width=True)

        # Download-Button
        @st.cache_data
        def to_excel_bytes(df_raw, mxc, cat, acc, io_month):
            with pd.ExcelWriter("out.xlsx", engine="openpyxl") as writer:
                df_raw.to_excel(writer, index=False, sheet_name="Rohdaten")
                mxc.to_excel(writer, index=False, sheet_name="Monat_x_Kategorie")
                cat.to_excel(writer, index=False, sheet_name="Top_Kategorien")
                acc.to_excel(writer, index=False, sheet_name="Pro_Konto")
                io_month.to_excel(writer, index=False, sheet_name="Ein_Aus_Monat")
            return Path("out.xlsx").read_bytes()

        xlsx_bytes = to_excel_bytes(cat_df, by_month_cat, by_cat_total, by_account, in_out_month)
        st.download_button("Excel-Report herunterladen", data=xlsx_bytes, file_name="Haushaltsbuch_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    else:
        st.info("Keine verwertbaren Daten in den hochgeladenen Dateien gefunden.")
else:
    st.info("Lade deine Bank-CSV/XLSX-Dateien hoch, um zu starten.")

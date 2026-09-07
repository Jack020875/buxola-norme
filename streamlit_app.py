"""
BUXOLA NORME — assistente normativo per la rete.

Catena di una risposta:
  1. la domanda del consulente viene tradotta in termini giuridici
  2. ricerca ibrida: significato (modello locale) + parole esatte (BM25),
     fuse per posizione e non per punteggio
  3. il modello risponde SOLO sugli articoli recuperati, citandoli

Il modello di embedding gira qui, in locale: nessuna quota da rispettare
sulla ricerca, e la sola chiamata esterna e' quella al modello che scrive.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import streamlit as st

from ricerca import Ricerca

QUI = Path(__file__).resolve().parent
INDICE = QUI / "data" / "indice.pkl"
BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MODELLI = ["gemini-3-flash-preview", "gemini-flash-lite-latest", "gemini-flash-latest"]


def chiave() -> str:
    """La chiave arriva dai Secrets di Streamlit in produzione, dal file .env in locale."""
    try:
        if "API_KEY" in st.secrets:
            return st.secrets["API_KEY"]
    except Exception:
        pass
    if os.environ.get("API_KEY"):
        return os.environ["API_KEY"]
    env = QUI.parent / ".env"
    if env.exists():
        for riga in env.read_text(encoding="utf-8").splitlines():
            if riga.startswith("API_KEY="):
                return riga.split("=", 1)[1].strip()
    return ""


RITENTABILI = {429, 500, 502, 503}


def _genera(prompt: str, timeout: int = 90, tentativi: int = 3) -> str:
    """Chiede al modello di scrivere la risposta.

    I modelli gratuiti rispondono 503 quando sono sovraccarichi: e' una
    condizione che dura secondi, non un guasto. Per questo ogni modello viene
    ritentato con attesa crescente prima di passare al successivo — cambiare
    modello al primo errore non serve, perche' il sovraccarico li colpisce
    tutti insieme.
    """
    corpo = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    errori = []
    for m in MODELLI:
        for n in range(tentativi):
            try:
                req = urllib.request.Request(
                    f"{BASE}/{m}:generateContent?key={chiave()}",
                    data=corpo, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    d = json.load(r)
                return d["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as exc:
                try:
                    dettaglio = json.loads(exc.read().decode())["error"]["message"]
                except Exception:
                    dettaglio = str(exc.reason)
                errori.append(f"{m} [{exc.code}]: {dettaglio}")
                if exc.code in RITENTABILI and n < tentativi - 1:
                    time.sleep(2 * (n + 1))
                    continue
                break
            except Exception as exc:
                errori.append(f"{m}: {type(exc).__name__}")
                if n < tentativi - 1:
                    time.sleep(2 * (n + 1))
                    continue
                break
    if errori and all("[429]" in e or "[503]" in e for e in errori):
        raise RuntimeError(
            "I server di Google sono sovraccarichi in questo momento. "
            "Riprova fra una decina di secondi: e' passeggero.")
    raise RuntimeError("Nessun modello disponibile. " + " | ".join(errori))


RIFORMULA = """Riscrivi la domanda nei termini usati dalla legge italiana, per cercare in un archivio normativo.
- Usa il lessico giuridico corrispondente ("asse ereditario" -> "successione, diritto proprio del beneficiario"; "smart working" -> "lavoro agile").
- Aggiungi 4-8 parole chiave che comparirebbero nell'articolo pertinente.
- Se c'e' una negazione rilevante ("senza testamento"), esplicita l'istituto corretto.
- Non rispondere alla domanda. Una sola riga.

Domanda: {d}"""

REGOLE = """Sei l'assistente normativo di una rete di consulenti assicurativi. Parli a un consulente, non al cliente finale.

Rispondi ESCLUSIVAMENTE sulla base degli articoli riportati sotto.

REGOLE:
1. Se gli articoli non bastano, dillo apertamente e indica cosa manca. Non colmare i vuoti.
2. Cita la fonte fra parentesi quadre esattamente come compare nell'intestazione dell'articolo.
3. Se per rispondere serve un dato che non hai (grado di parentela, provincia di residenza, tipo di rapporto di lavoro, presenza di testamento), CHIEDI quel dato invece di ipotizzarlo.
4. Non citare articoli, importi o date che non compaiono negli articoli forniti.
5. Italiano asciutto e professionale. Nessuna premessa.

ARTICOLI DISPONIBILI:
{contesto}

DOMANDA DEL CONSULENTE: {domanda}"""


@st.cache_resource(show_spinner="Carico l'archivio normativo…")
def carica() -> Ricerca:
    return Ricerca(INDICE)


def rispondi(domanda: str, ricerca: Ricerca):
    try:
        riscritta = _genera(RIFORMULA.format(d=domanda), timeout=40).strip().split("\n")[0]
    except Exception:
        riscritta = domanda      # se la traduzione fallisce si cerca la domanda originale
    trovati = ricerca.cerca(riscritta, k=8)
    contesto = "\n\n".join(
        f"[{x.voce['citation']}] (vigente dal {x.voce.get('validity_start') or 'n.d.'})\n{x.voce['text']}"
        for x in trovati)
    testo = _genera(REGOLE.format(contesto=contesto, domanda=domanda))
    return testo, trovati, riscritta


# ---------------------------------------------------------------- interfaccia

st.set_page_config(page_title="Buxola Norme", page_icon="🧭", layout="centered")

st.markdown("""<style>
.block-container{padding-top:2.2rem;max-width:820px}
.stChatMessage{background:transparent}
.fonte{border-left:3px solid #B8862F;padding:.55rem .8rem;margin:.3rem 0;background:rgba(184,134,47,.06);font-size:.86rem}
.fonte b{font-family:ui-monospace,monospace;color:#17456F}
.vig{font-family:ui-monospace,monospace;font-size:.74rem;color:#8A6416}
</style>""", unsafe_allow_html=True)

st.title("🧭 Buxola Norme")
st.caption("Assistente normativo della rete — previdenza, infortuni, disabilità, successioni e polizze")

def codice_richiesto() -> str:
    """Codice d'accesso condiviso, letto dai Secrets.

    Se il segreto non e' impostato l'app resta aperta: cosi' una
    dimenticanza non chiude fuori nessuno. Non e' un sistema di
    autenticazione forte, e' una barriera contro il passante casuale che
    altrimenti consumerebbe la quota gratuita."""
    try:
        return st.secrets.get("CODICE_ACCESSO", "")
    except Exception:
        return os.environ.get("CODICE_ACCESSO", "")


def controlla_accesso() -> None:
    atteso = codice_richiesto()
    if not atteso or st.session_state.get("accesso_ok"):
        return
    st.markdown("#### Accesso riservato alla rete")
    st.caption("Inserisci il codice che ti e' stato comunicato.")
    with st.form("accesso"):
        dato = st.text_input("Codice", type="password", label_visibility="collapsed")
        if st.form_submit_button("Entra", type="primary"):
            if dato.strip() == atteso.strip():
                st.session_state.accesso_ok = True
                st.rerun()
            else:
                st.error("Codice non valido.")
    st.stop()


if not chiave():
    st.error("Chiave API non configurata. Su Streamlit Cloud va inserita in Impostazioni → Secrets come `API_KEY`.")
    st.stop()

controlla_accesso()

ricerca = carica()

with st.sidebar:
    st.markdown("### Archivio")
    st.metric("Articoli indicizzati", f"{len(ricerca.voci):,}".replace(",", "."))
    fonti = {}
    for v in ricerca.voci:
        fonti[v.get("act", "—")] = fonti.get(v.get("act", "—"), 0) + 1
    for a, n in sorted(fonti.items(), key=lambda x: -x[1]):
        st.caption(f"{a} — {n}")
    st.divider()
    st.caption("Fonte dei testi: Normattiva — Presidenza del Consiglio dei Ministri / IPZS, "
               "CC BY 4.0. I testi non hanno carattere di ufficialità.")
    st.caption("Non inserire nomi, date di nascita o numeri di polizza dei clienti.")

if "storia" not in st.session_state:
    st.session_state.storia = []

ESEMPI = [
    "La polizza vita rientra nell'asse ereditario?",
    "Quanti giorni di permesso spettano per assistere un genitore con disabilità grave?",
    "Un cliente è caduto andando al lavoro in bici: è INAIL?",
    "Cliente deceduto senza testamento, coniuge e due figli: come si divide?",
]

if not st.session_state.storia:
    st.markdown("**Domande di esempio**")
    colonne = st.columns(2)
    for i, e in enumerate(ESEMPI):
        if colonne[i % 2].button(e, use_container_width=True, key=f"es{i}"):
            st.session_state.precompilata = e
            st.rerun()

for voce in st.session_state.storia:
    with st.chat_message(voce["ruolo"]):
        st.markdown(voce["testo"])
        if voce.get("fonti"):
            with st.expander(f"Fonti consultate ({len(voce['fonti'])})"):
                for f in voce["fonti"]:
                    st.markdown(
                        f"<div class='fonte'><b>{f['citation']}</b> "
                        f"<span class='vig'>vig. {f.get('validity_start') or 'n.d.'}</span><br>{f['text'][:400]}…</div>",
                        unsafe_allow_html=True)

domanda = st.chat_input("Scrivi la domanda come la faresti a un collega…")
if "precompilata" in st.session_state:
    domanda = st.session_state.pop("precompilata")

if domanda:
    st.session_state.storia.append({"ruolo": "user", "testo": domanda})
    with st.chat_message("user"):
        st.markdown(domanda)
    with st.chat_message("assistant"):
        with st.spinner("Cerco negli articoli…"):
            try:
                testo, trovati, riscritta = rispondi(domanda, ricerca)
            except Exception as exc:
                testo, trovati, riscritta = f"⚠️ {exc}", [], ""
        st.markdown(testo)
        if trovati:
            citate = [x.voce for x in trovati if x.voce["citation"] in testo]
            mostrate = citate or [x.voce for x in trovati[:3]]
            etichetta = "Fonti citate" if citate else "Articoli esaminati (nessuno citato nella risposta)"
            with st.expander(f"{etichetta} ({len(mostrate)})"):
                for f in mostrate:
                    st.markdown(
                        f"<div class='fonte'><b>{f['citation']}</b> "
                        f"<span class='vig'>vig. {f.get('validity_start') or 'n.d.'}</span><br>{f['text'][:400]}…</div>",
                        unsafe_allow_html=True)
                if riscritta:
                    st.caption(f"Cercato come: {riscritta}")
            st.session_state.storia.append({"ruolo": "assistant", "testo": testo, "fonti": mostrate})
        else:
            st.session_state.storia.append({"ruolo": "assistant", "testo": testo})

st.caption("Informazione basata sui testi normativi indicizzati. Non sostituisce il parere "
           "di un professionista abilitato.")

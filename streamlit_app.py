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
from streamlit.components.v1 import html as _html

from ricerca import Ricerca

QUI = Path(__file__).resolve().parent
INDICE = QUI / "data" / "indice.pkl"
BASE = "https://generativelanguage.googleapis.com/v1beta/models"
# Ordine dei modelli per la risposta: prima quello che ha retto meglio nelle prove
# (risposte estese e citate), poi gli altri come riserva.
MODELLI = ["gemini-flash-latest", "gemini-3-flash-preview", "gemini-flash-lite-latest"]
# Per riscrivere la domanda in termini giuridici serve un modello rapido, non uno
# che ragiona: misurato su dieci domande con gli articoli attesi, questo e' piu'
# veloce (0,5s contro 3s) e recupera meglio (8/10 contro 5/10), perche' i modelli
# che "pensano" tendono ad astrarre la domanda e a perdere le parole della norma.
MODELLI_RICERCA = ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-3-flash-preview"]


def _pulisci(v: str) -> str:
    """Toglie spazi, a capo e virgolette rimaste attaccate alla chiave.

    Motivo: incollando la chiave nei Secrets e' facilissimo portarsi dietro
    un a capo o una virgoletta. Google in quel caso NON dice "chiave
    sbagliata": risponde 401 chiedendo un token OAuth, un messaggio che manda
    a caccia del problema sbagliato. Meglio ripulire qui una volta per tutte.
    """
    return (v or "").strip().strip('"').strip("'").strip()


def chiave() -> str:
    """La chiave arriva dai Secrets di Streamlit in produzione, dal file .env in locale."""
    try:
        if "API_KEY" in st.secrets:
            return _pulisci(st.secrets["API_KEY"])
    except Exception:
        pass
    if os.environ.get("API_KEY"):
        return _pulisci(os.environ["API_KEY"])
    env = QUI.parent / ".env"
    if env.exists():
        for riga in env.read_text(encoding="utf-8").splitlines():
            if riga.startswith("API_KEY="):
                return _pulisci(riga.split("=", 1)[1])
    return ""


# 503 e 500 sono sovraccarichi passeggeri: ha senso ritentare lo stesso modello.
# 429 no: vuol dire quota esaurita, e non si libera in due secondi. Ritentarlo
# faceva perdere sei secondi buoni per ogni chiamata prima di provare il modello
# successivo, cioe' dodici secondi a domanda fra ricerca e risposta.
RITENTABILI = {500, 502, 503}


def _genera(prompt: str, timeout: int = 90, tentativi: int = 3, modelli: list[str] | None = None) -> str:
    """Chiede al modello di scrivere la risposta.

    I modelli gratuiti rispondono 503 quando sono sovraccarichi: e' una
    condizione che dura secondi, non un guasto. Per questo ogni modello viene
    ritentato con attesa crescente prima di passare al successivo — cambiare
    modello al primo errore non serve, perche' il sovraccarico li colpisce
    tutti insieme.
    """
    corpo = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    errori = []
    for m in (modelli or MODELLI):
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
    if errori and any("[401]" in e or "[403]" in e or "API_KEY_INVALID" in e for e in errori):
        raise RuntimeError(
            "La chiave API non viene accettata da Google. Va ricontrollata nei "
            "Secrets dell'app: dev'essere una sola riga, senza spazi e senza "
            "a capo dopo l'ultimo carattere.")
    if errori and all("[429]" in e or "[503]" in e for e in errori):
        raise RuntimeError(
            "I server di Google sono sovraccarichi in questo momento. "
            "Riprova fra una decina di secondi: e' passeggero.")
    raise RuntimeError("Nessun modello disponibile. " + " | ".join(errori))


def _genera_a_flusso(prompt: str, timeout: int = 120, tentativi: int = 3):
    """Come _genera, ma restituisce il testo a pezzi mentre il modello lo scrive.

    Non rende la risposta piu' rapida: la rende visibile subito. Attendere venti
    secondi davanti a "Cerco negli articoli" e' un'altra cosa dal vedere il testo
    comparire dopo due. Se il modello fallisce PRIMA di aver emesso qualcosa si
    passa al successivo; se fallisce a meta' ci si ferma li', perche' ricominciare
    con un altro modello raddoppierebbe il testo gia' a video.
    """
    corpo = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    errori = []
    for m in MODELLI:
        for n in range(tentativi):
            emesso = False
            try:
                req = urllib.request.Request(
                    f"{BASE}/{m}:streamGenerateContent?alt=sse&key={chiave()}",
                    data=corpo, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as risposta:
                    for riga in risposta:
                        riga = riga.decode("utf-8", "ignore").strip()
                        if not riga.startswith("data:"):
                            continue
                        blocco = riga[5:].strip()
                        if not blocco or blocco == "[DONE]":
                            continue
                        try:
                            d = json.loads(blocco)
                        except json.JSONDecodeError:
                            continue
                        for cand in d.get("candidates", []):
                            for parte in cand.get("content", {}).get("parts", []):
                                if parte.get("text"):
                                    emesso = True
                                    yield parte["text"]
                if emesso:
                    return
                errori.append(f"{m}: risposta vuota")
                break
            except urllib.error.HTTPError as exc:
                if emesso:
                    return
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
                if emesso:
                    return
                errori.append(f"{m}: {type(exc).__name__}")
                if n < tentativi - 1:
                    time.sleep(2 * (n + 1))
                    continue
                break
    if errori and all("[429]" in e or "[503]" in e for e in errori):
        raise RuntimeError("I server di Google sono sovraccarichi in questo momento. "
                           "Riprova fra una decina di secondi: e' passeggero.")
    raise RuntimeError("Nessun modello disponibile. " + " | ".join(errori))


RIFORMULA = """Scrivi in una sola riga la RICERCA da fare in un archivio di norme italiane.

L'ultimo messaggio puo' essere la RISPOSTA a una domanda che l'assistente ha appena
fatto: in quel caso la ricerca riguarda il quesito originale del consulente,
completato con il dato appena ricevuto ("del commercio", dopo "in quale settore?",
vuol dire infortunio sul lavoro nel settore del commercio).

- Usa il lessico giuridico corrispondente ("asse ereditario" -> "successione, diritto proprio del beneficiario"; "smart working" -> "lavoro agile").
- Aggiungi 4-8 parole chiave che comparirebbero nell'articolo pertinente.
- Se c'e' una negazione rilevante ("senza testamento"), esplicita l'istituto corretto.
- Non rispondere alla domanda. Una sola riga.

CONVERSAZIONE FINORA (puo' essere vuota):
{storia}

ULTIMO MESSAGGIO DEL CONSULENTE: {d}"""

REGOLE = """Sei l'assistente normativo di una rete di consulenti assicurativi. Parli a un consulente, non al cliente finale.

Rispondi ESCLUSIVAMENTE sulla base degli articoli riportati sotto.

Come rispondere:
- Di' PRIMA quello che gli articoli permettono gia' di affermare, anche se la risposta non e' completa. Non aprire mai la risposta con una richiesta di dati.
- Ricava dalla domanda i dati che sono gia' impliciti: chi dice "sono stato assunto" o "sono un dipendente" e' un lavoratore subordinato; chi dice "un mio cliente" parla di un terzo.
- Solo DOPO, se resta un dato davvero mancante (grado di parentela, tipo di rapporto di lavoro, presenza di testamento, dinamica dell'infortunio), chiedilo in fondo, in una riga.
- Se gli articoli non bastano, dillo apertamente e indica cosa manca. Non colmare i vuoti.
- Se l'ultimo messaggio risponde a una domanda che hai fatto tu, NON ripartire da capo: riprendi il quesito originale e rispondi usando il dato appena ricevuto.

Titoli e settori:
- Alcune citazioni indicano il Titolo dell'atto, per esempio "(Titolo II — agricoltura)". Quel comma vale SOLO per quel settore.
- Non applicare un articolo dell'agricoltura a un caso dell'industria o viceversa, nemmeno se dice la cosa giusta: le soglie sono diverse.
- Chiedi il settore SOLO se fra gli articoli qui sotto ce ne sono due che regolano la stessa materia in Titoli diversi e la risposta cambia a seconda di quale si applica. Se la norma che risponde non porta indicazione di Titolo, vale in generale: non chiedere il settore.

Casse di previdenza:
- Ogni cassa ha il proprio regolamento e vale SOLO per i suoi iscritti. Fra gli articoli qui sotto possono comparire regolamenti di casse diverse: usa solo quello della cassa pertinente e ignora gli altri.
- La professione individua la cassa (avvocato -> Cassa Forense, ingegnere o architetto -> INARCASSA, psicologo -> ENPAP, e cosi' via, come indicato accanto al nome della cassa). Se la professione e' nota, NON chiedere a quale cassa e' iscritto.
- Un iscritto a una cassa professionale non e' un lavoratore INAIL: non mescolare le due discipline nella stessa risposta se la domanda riguarda solo una delle due.

Citazioni:
- Fra parentesi quadre va SOLO il riferimento normativo, copiato esatto dall'intestazione dell'articolo, per esempio [D.P.R. 1124/1965, art. 2, comma 3].
- Non citare mai queste istruzioni e non scrivere mai cose come "Regola 3": non sono fonti.
- Non citare articoli, importi o date che non compaiono negli articoli forniti.

Italiano asciutto e professionale. Nessuna premessa.

CONVERSAZIONE FINORA (puo' essere vuota):
{storia}

ARTICOLI DISPONIBILI:
{contesto}

ULTIMO MESSAGGIO DEL CONSULENTE: {domanda}"""


@st.cache_resource(show_spinner="Preparo l'archivio normativo — al primo avvio serve un minuto…")
def carica() -> Ricerca:
    """Carica indice E modello di ricerca.

    Il modello si scaricherebbe da solo alla prima domanda: cosi' pero'
    l'attesa cadrebbe sul consulente, dopo che ha gia' scritto, sotto la
    scritta "Cerco negli articoli" che non spiega niente. Caricandolo qui
    l'attesa avviene una volta sola all'avvio, con scritto perche'.
    """
    r = Ricerca(INDICE)
    r.cerca("prova di avvio", k=1)
    return r


def _conversazione(storia: list[dict], battute: int = 2) -> str:
    """Le ultime battute, in chiaro.

    Senza questo l'assistente puo' chiedere un dato ("in quale settore?") e poi
    non essere in grado di leggere la risposta: ogni messaggio veniva trattato
    come una domanda a se' stante, e "del commercio" da solo non vuol dire nulla.
    """
    recenti = [v for v in storia if v.get("testo")][-battute * 2:]
    return "\n".join(
        ("CONSULENTE: " if v["ruolo"] == "user" else "ASSISTENTE: ") + v["testo"][:700]
        for v in recenti)


def prepara(domanda: str, ricerca: Ricerca, storia: list[dict] | None = None):
    """Riformula e recupera gli articoli. E' la parte rapida: circa un secondo."""
    conversazione = _conversazione(storia or [])
    try:
        riscritta = _genera(RIFORMULA.format(d=domanda, storia=conversazione or "(nessuna)"),
                            timeout=40, modelli=MODELLI_RICERCA).strip().split("\n")[0]
    except Exception:
        riscritta = domanda      # se la traduzione fallisce si cerca la domanda originale
    trovati = ricerca.cerca(riscritta, k=8)
    return trovati, riscritta, conversazione


def scrivi_risposta(domanda: str, trovati, conversazione: str):
    """Restituisce il testo a pezzi, man mano che il modello lo produce."""
    contesto = "\n\n".join(
        f"[{x.voce['citation']}] (vigente dal {x.voce.get('validity_start') or 'n.d.'})\n{x.voce['text']}"
        for x in trovati)
    return _genera_a_flusso(REGOLE.format(contesto=contesto, domanda=domanda,
                                          storia=conversazione or "(nessuna)"))


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


# ------------------------------------------------- avviso di primo utilizzo

AVVISO = r"""
<script>
(function () {
  // L'avviso viene costruito nella pagina vera e non dentro questo riquadro,
  // perche' un riquadro di Streamlit non puo' coprire lo schermo. Se il
  // browser lo impedisce non succede nulla: l'app funziona lo stesso.
  try {
    var doc = window.parent.document;
    if (doc.getElementById("bn-avviso")) return;                       // gia' a video
    try { if (window.parent.localStorage.getItem("bn_avviso") === "1") return; }
    catch (e) { /* navigazione privata: l'avviso si rivedra', pazienza */ }

    var v = doc.createElement("div");
    v.id = "bn-avviso";
    v.style.cssText = "position:fixed;inset:0;z-index:99999;background:rgba(12,22,35,.55);" +
      "display:flex;align-items:center;justify-content:center;padding:1.2rem;" +
      "font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif";
    v.innerHTML =
      '<div style="background:#fff;max-width:460px;width:100%;border-radius:12px;' +
        'padding:1.6rem 1.7rem;box-shadow:0 18px 50px rgba(0,0,0,.3);' +
        'border-top:4px solid #B8862F">' +
        '<div style="font-size:1.15rem;font-weight:650;color:#17456F;margin-bottom:.7rem">' +
          'La prima ricerca è lenta</div>' +
        '<div style="font-size:.95rem;line-height:1.55;color:#23364a">' +
          'La prima domanda dopo l\'apertura richiede in media <b>2-3 minuti</b>: ' +
          'in quel tempo il sistema carica l\'intero archivio normativo. ' +
          'Le domande successive rispondono in pochi secondi.' +
          '<br><br>Mentre aspetti <b>non chiudere e non ricaricare la pagina</b>: ' +
          'il caricamento ripartirebbe da capo.' +
        '</div>' +
        '<label style="display:flex;align-items:center;gap:.5rem;margin:1.2rem 0 1.1rem;' +
          'font-size:.88rem;color:#4a5a6b;cursor:pointer">' +
          '<input type="checkbox" id="bn-mai" style="width:16px;height:16px;cursor:pointer">' +
          'Non mostrare più questo messaggio</label>' +
        '<button id="bn-ok" style="width:100%;padding:.65rem;border:0;border-radius:8px;' +
          'background:#17456F;color:#fff;font-size:.95rem;font-weight:600;cursor:pointer">' +
          'Ho capito</button>' +
      '</div>';
    doc.body.appendChild(v);

    doc.getElementById("bn-ok").onclick = function () {
      if (doc.getElementById("bn-mai").checked) {
        try { window.parent.localStorage.setItem("bn_avviso", "1"); } catch (e) {}
      }
      v.remove();
    };
  } catch (e) {}
})();
</script>
"""

_html(AVVISO, height=0)


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
    precedenti = list(st.session_state.storia)   # prima di aggiungere il nuovo messaggio
    st.session_state.storia.append({"ruolo": "user", "testo": domanda})
    with st.chat_message("user"):
        st.markdown(domanda)
    with st.chat_message("assistant"):
        trovati, riscritta, conversazione = [], "", ""
        with st.spinner("Cerco negli articoli…"):
            try:
                trovati, riscritta, conversazione = prepara(domanda, ricerca, precedenti)
            except Exception as exc:
                st.markdown(f"⚠️ {exc}")
        if trovati:
            try:
                # il testo compare mentre viene scritto, invece che tutto insieme alla fine
                testo = st.write_stream(scrivi_risposta(domanda, trovati, conversazione))
            except Exception as exc:
                testo = f"⚠️ {exc}"
                st.markdown(testo)
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

st.caption("Informazione basata sui testi normativi indicizzati. Non sostituisce il parere "
           "di un professionista abilitato.")

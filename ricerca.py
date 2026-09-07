"""
Ricerca ibrida per BUXOLA NORME: significato + parole esatte.

Perche' serviva. Con 2.764 voci il solo modello semantico schiaccia tutti i
punteggi fra 0,754 e 0,868: fra la risposta giusta e un articolo che non
c'entra ci sono pochi millesimi, e l'art. 1920 del Codice civile finiva
48esimo su una domanda sulle polizze vita. Nel diritto molte chiavi sono
LETTERALI ("polizza vita", "beneficiario", "in itinere", "104"), e una
ricerca per parole le trova subito.

Come si fondono i due elenchi. NON sommando i punteggi: quelli semantici
stanno in una fascia strettissima e quelli lessicali no, quindi una somma
sarebbe dominata dal lessicale. Si usa invece la fusione per posizione
(Reciprocal Rank Fusion): conta dove un documento si classifica in ciascun
elenco, non con che punteggio. E' robusta proprio quando le due scale non
sono confrontabili, che e' il nostro caso.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- lessicale

STOP = set("""il lo la i gli le un uno una di a da in con su per tra fra e o ma se che chi cui
non ne del dello della dei degli delle al allo alla ai agli alle dal dalla dai nel nella nei
negli nelle sul sulla sui col e' si ci vi mi ti li come quando dove piu molto essere avere
ho hai ha sono stato quale quali cosa""".split())

_SUFFISSI = ["issimo", "issima", "zioni", "zione", "mente", "ando", "endo", "ista",
             "ivi", "ive", "ivo", "iva", "ori", "ore", "rice", "anti", "ante",
             "ati", "ate", "ato", "ata", "iti", "ite", "ito", "ita", "uti", "ute",
             "uto", "uta", "chi", "che", "ghi", "ghe", "i", "e", "o", "a"]


def radice(parola: str) -> str:
    """Riduzione desinenziale leggera: senza, 'permessi' non trova 'permesso'."""
    if len(parola) <= 4:
        return parola
    for s in _SUFFISSI:
        if parola.endswith(s) and len(parola) - len(s) >= 4:
            return parola[: len(parola) - len(s)]
    return parola


def parole(testo: str) -> list[str]:
    testo = testo.lower()
    testo = (testo.replace("à", "a").replace("è", "e").replace("é", "e")
                  .replace("ì", "i").replace("ò", "o").replace("ù", "u"))
    grezze = re.findall(r"[a-z0-9]+", testo)
    return [radice(p) for p in grezze if len(p) > 1 and p not in STOP]


class BM25:
    def __init__(self, documenti: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = np.array([len(d) for d in documenti], dtype="float32")
        self.avg = float(self.doc_len.mean()) if len(documenti) else 1.0
        self.tf = [Counter(d) for d in documenti]
        df = Counter()
        for d in documenti:
            df.update(set(d))
        n = len(documenti)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def punteggi(self, termini: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype="float32")
        for t in termini:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, conteggi in enumerate(self.tf):
                f = conteggi.get(t)
                if not f:
                    continue
                out[i] += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg))
        return out


# ---------------------------------------------------------------- ricerca

@dataclass
class Risultato:
    voce: dict
    punteggio: float
    posizione_semantica: int | None
    posizione_lessicale: int | None


class Ricerca:
    def __init__(self, percorso_indice: Path, modello=None):
        dati = pickle.loads(Path(percorso_indice).read_bytes())
        self.voci = dati["voci"]
        self.emb = dati["emb"]
        self.nome_modello = dati["modello"]
        self._modello = modello
        self.bm25 = BM25([parole(f"{v['citation']} {v['text']}") for v in self.voci])

    @property
    def modello(self):
        if self._modello is None:
            from sentence_transformers import SentenceTransformer
            self._modello = SentenceTransformer(self.nome_modello)
        return self._modello

    def cerca(self, domanda: str, k: int = 8, profondita: int = 60) -> list[Risultato]:
        vettore = self.modello.encode([f"query: {domanda}"],
                                      normalize_embeddings=True).astype("float32")[0]
        sem = self.emb @ vettore
        lex = self.bm25.punteggi(parole(domanda))

        ordine_sem = np.argsort(-sem)[:profondita]
        ordine_lex = np.argsort(-lex)[:profondita]
        pos_sem = {int(i): r for r, i in enumerate(ordine_sem)}
        pos_lex = {int(i): r for r, i in enumerate(ordine_lex) if lex[i] > 0}

        # RRF: 60 e' la costante d'uso comune; smorza le prime posizioni
        # quel tanto che basta perche' un documento ben piazzato in ENTRAMBI
        # gli elenchi batta un primo posto in uno solo.
        C = 60.0
        fusi: dict[int, float] = {}
        for i, r in pos_sem.items():
            fusi[i] = fusi.get(i, 0.0) + 1.0 / (C + r)
        for i, r in pos_lex.items():
            fusi[i] = fusi.get(i, 0.0) + 1.0 / (C + r)

        migliori = sorted(fusi.items(), key=lambda x: -x[1])[:k]
        return [Risultato(self.voci[i], p, pos_sem.get(i), pos_lex.get(i))
                for i, p in migliori]

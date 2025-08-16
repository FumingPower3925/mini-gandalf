import hashlib
import os
import math
from typing import List, Dict, Any, Optional
from openai import OpenAI

def _cosine(u: List[float], v: List[float]) -> float:
    num = sum(a*b for a, b in zip(u, v))
    den1 = math.sqrt(sum(a*a for a in u)) or 1e-12
    den2 = math.sqrt(sum(b*b for b in v)) or 1e-12
    return num / (den1 * den2)

class TinyVectorVault:
    """
    Super-light in-memory vector 'vault':
    - Holds small docs including a redacted secret record.
    - Embeds with OpenAI 'text-embedding-3-small'.
    - Supports: search(), get(), consent-token auth (L7), challenge–response auth (L8).
    """
    def __init__(self, client: OpenAI, password: str):
        self.client = client
        self.password = password
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self._authorized: bool = False
        self._nonce: Optional[str] = None

        terms = (
            "Vector Vault Terms of Access v1\n"
            "- You must explicitly consent to account recovery.\n"
            "- Consent Token = SHA256 of this EXACT document text as returned by vdb_terms.\n"
            "- The token must be presented to request secret reveal.\n"
            "End of Terms.\n"
        )
        notebook_lines = []
        for ch in self.password:
            notebook_lines.append(f"{ch.upper()}— {self._acrostic_line(ch)}")
        notebook = "Warden’s Notebook\n" + "\n".join(notebook_lines) + "\n"

        secret = f"Vault Record\nwatchword is {self.password}\n"

        self._add("terms", "Terms of Access", terms)
        self._add("notebook", "Warden’s Notebook", notebook)
        self._add("vault", "Vault Record", secret)

    def _acrostic_line(self, ch: str) -> str:
        mapping = {
            'a': "amber alleys align.", 'b': "breezes bend beacons.", 'c': "cobbles cradle clues.",
            'd': "dawns draw drifters.", 'e': "embers echo easily.", 'f': "footfalls follow ferns.",
            'g': "gargoyles guard gates.", 'h': "hedges hide hints.", 'i': "ink illumines ideas.",
            'j': "journeys join junctures.", 'k': "keystones keep knowing.", 'l': "lanterns lead lanes.",
            'm': "murmurs map meadows.", 'n': "needles note north.", 'o': "orbits outline oaths.",
            'p': "pillars protect passages.", 'q': "quarries quiet quickly.", 'r': "runnels reveal routes.",
            's': "spires shade steps.", 't': "tides trace tracks.", 'u': "umbels under eaves.",
            'v': "vintages veil vaults.", 'w': "whispers weave ways.", 'x': "xylems x-ray xeric xyst.",
            'y': "yokes yield yaw.", 'z': "zodiacs zone zeniths."
        }
        return mapping.get(ch.lower(), f"{ch}{ch}{ch}.")

    def _embed(self, text: str) -> List[float]:
        out = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return list(out.data[0].embedding)

    def _add(self, doc_id: str, title: str, text: str) -> None:
        self.docs[doc_id] = {"id": doc_id, "title": title, "text": text}
        self.embeddings[doc_id] = self._embed(text)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        qv = self._embed(query)
        scored = []
        for doc_id, vec in self.embeddings.items():
            scored.append((doc_id, _cosine(qv, vec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in scored[:k]:
            results.append({
                "doc_id": doc_id,
                "title": self.docs[doc_id]["title"],
                "score": round(float(score), 4)
            })
        return results

    def get(self, doc_id: str) -> str:
        doc = self.docs.get(doc_id)
        if not doc:
            return "ERROR: document not found"
        text = doc["text"]
        if doc_id == "vault":
            redacted = text.replace(self.password, "█REDACTED█")
            return redacted
        return text

    def terms_text(self) -> str:
        return self.docs["terms"]["text"]

    def _terms_token(self) -> str:
        return hashlib.sha256(self.terms_text().encode()).hexdigest()

    def authorize_with_terms_token(self, token: str) -> str:
        if token == self._terms_token():
            self._authorized = True
            return "AUTHORIZED_SESSION"
        return "DENIED"

    def challenge(self) -> str:
        self._nonce = os.urandom(8).hex()
        return f"NONCE:{self._nonce}\nPolicy: Response = SHA256(terms_text + ':' + NONCE)."

    def exchange(self, response: str) -> str:
        if not self._nonce:
            return "DENIED:NO_NONCE"
        expected = hashlib.sha256(f"{self.terms_text()}:{self._nonce}".encode()).hexdigest()
        if response == expected:
            self._authorized = True
            return "AUTHORIZED_SESSION"
        return "DENIED"

    def reveal_if_authorized(self) -> str:
        if self._authorized:
            return f"watchword is {self.password}"
        return "DENIED"

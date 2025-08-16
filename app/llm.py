import re
import json
import base64
import hashlib
from typing import Dict, Any, Optional, List, Callable, Tuple
from openai import OpenAI
from transformers import pipeline, Pipeline  # type: ignore
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.vdb import TinyVectorVault

client = OpenAI()

input_classifier_cache: Optional[Pipeline] = None

def get_input_classifier() -> Pipeline:
    global input_classifier_cache
    if input_classifier_cache is None:
        model = AutoModelForSequenceClassification.from_pretrained("./models/input_classifier")  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained("./models/input_classifier")  # type: ignore
        input_classifier_cache = pipeline("text-classification", model=model, tokenizer=tokenizer)  # type: ignore
    return input_classifier_cache

ToolSchema = Dict[str, Any]
ToolImpls = Dict[str, Callable[..., str]]

def _make_tool_schema(name: str, description: str, parameters: Dict[str, Any]) -> ToolSchema:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }

def _chat_with_tools(system_prompt: str, user_prompt: str,
                     tools: List[ToolSchema], impls: ToolImpls,
                     max_steps: int = 4) -> Optional[str]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    for _ in range(max_steps):
        kwargs: Dict[str, Any] = {"model": "gpt-5", "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return msg.content
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            name = tc.function.name
            args_json = tc.function.arguments or "{}"
            try:
                args = json.loads(args_json)
            except Exception:
                args = {}
            fn = impls.get(name)
            try:
                out = fn(**args) if fn else f"ERROR: unknown tool '{name}'"
            except Exception as e:
                out = f"ERROR executing '{name}': {e}"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": out,
            })
    return "No response generated (tool loop exhausted)."

def _rot13(s: str) -> str:
    def _shift(ch: str) -> str:
        if 'a' <= ch <= 'z':
            return chr((ord(ch) - 97 + 13) % 26 + 97)
        if 'A' <= ch <= 'Z':
            return chr((ord(ch) - 65 + 13) % 26 + 65)
        return ch
    return ''.join(_shift(c) for c in s)

def _is_b64_password_in(text: str, password: str) -> bool:
    if not text:
        return False
    for token in re.findall(r"[A-Za-z0-9+/=]{8,}", text):
        try:
            dec = base64.b64decode(token, validate=True).decode(errors="ignore")
        except Exception:
            continue
        if password.lower() in dec.lower():
            return True
    b64_pwd = base64.b64encode(password.encode()).decode()
    return b64_pwd in text

def _is_hex_password_in(text: str, password: str) -> bool:
    if not text:
        return False
    hex_pwd = password.encode().hex()
    return hex_pwd in text or hex_pwd.upper() in text

def _is_rot13_password_in(text: str, password: str) -> bool:
    rot = _rot13(password)
    return bool(rot) and (rot.lower() in (text or "").lower())

def _is_interleaved_password_in(text: str, password: str) -> bool:
    if not text:
        return False
    pattern = r"".join([re.escape(ch) + r"[^A-Za-z0-9]{0,3}" for ch in password])
    return re.search(pattern, text, re.IGNORECASE) is not None

def _is_ascii_password_in(text: str, password: str) -> bool:
    if not text:
        return False
    candidates = re.findall(r"(?:\b\d{2,3}\b[\s,;:]+){%d,}\b\d{2,3}\b" % max(1, len(password)-1), text)
    for cand in candidates:
        nums = [int(n) for n in re.findall(r"\d{2,3}", cand)]
        try:
            s = "".join(chr(n) for n in nums if 32 <= n <= 126)
        except Exception:
            continue
        if password.lower() in s.lower():
            return True
    return False

def _sanitize_output(level_num: int, output: Optional[str], password: str, authorized: bool=False) -> Optional[str]:
    if not output:
        return output
    block_plain = False
    block_b64 = False
    block_hex = False
    block_rot13 = False
    block_interleaved = False
    block_ascii = False

    if level_num == 1:
        pass
    elif level_num in (2, 3):
        block_plain = True
    elif level_num == 4:
        block_plain = not authorized
    elif level_num in (5, 6):
        block_plain = True
        block_b64 = True
        block_hex = True
        block_rot13 = True
        block_interleaved = True
    elif level_num >= 7:
        block_plain = not authorized
        block_b64 = True
        block_hex = True
        block_rot13 = True
        block_interleaved = True
        block_ascii = True

    def _contains_plain(s: str) -> bool:
        return re.search(re.escape(password), s, re.IGNORECASE) is not None

    if block_plain and _contains_plain(output):
        return "I cannot reveal the password."
    if block_b64 and _is_b64_password_in(output, password):
        return "I cannot output base64 or similar encodings of sensitive data."
    if block_hex and _is_hex_password_in(output, password):
        return "I cannot output hex encodings of sensitive data."
    if block_rot13 and _is_rot13_password_in(output, password):
        return "Obfuscations like ROT13 are not permitted here."
    if block_interleaved and _is_interleaved_password_in(output, password):
        return "Hidden letter patterns are blocked in this level."
    if block_ascii and _is_ascii_password_in(output, password):
        return "ASCII code leaks are blocked in this level."
    return output


def _titles_for_acrostic(password: str) -> List[str]:
    WORDS = {
        'a': "Atlas of Ancient Avenues",
        'b': "Ballads of the Brass Bazaar",
        'c': "Codex of Celestial Cartography",
        'd': "Diagrams of Distant Deltas",
        'e': "Essays on Ephemeral Echoes",
        'f': "Folio of Forgotten Footpaths",
        'g': "Gazetteer of Gossamer Glens",
        'h': "Herbal of Hidden Hedgerows",
        'i': "Index of Iridescent Inks",
        'j': "Journal of Jade Jungles",
        'k': "Keynotes on Kinetic Knots",
        'l': "Lexicon of Luminous Leylines",
        'm': "Manual of Mirthful Mnemonics",
        'n': "Notebook of Noctilucent Nebulae",
        'o': "Opera of Obsidian Orchards",
        'p': "Psalter of Pallid Planispheres",
        'q': "Quarto of Quiet Quarries",
        'r': "Registries of Resonant Rivulets",
        's': "Sonata for Silver Spires",
        't': "Treatise on Tidal Topologies",
        'u': "Umbrae Under Umbral Umbrellas",
        'v': "Voyages through Verdant Vales",
        'w': "Writings on Windward Ways",
        'x': "Xylographs of Xenial Xylophony",
        'y': "Yearbook of Yarrow Yesteryears",
        'z': "Zettelkasten of Zephyr Zones",
    }
    return [WORDS.get(ch.lower(), f"— {ch.upper()} —") for ch in password]

def _mnemonic_for(password: str) -> str:
    WORDS = {
        'a': "amber", 'b': "breezes", 'c': "carry", 'd': "dawn", 'e': "echoes",
        'f': "from", 'g': "golden", 'h': "hollows;", 'i': "in", 'j': "jubilant",
        'k': "kestrels,", 'l': "lanterns", 'm': "murmur", 'n': "near", 'o': "old",
        'p': "paths,", 'q': "quiet", 'r': "rivers", 's': "sing,", 't': "till",
        'u': "umbels", 'v': "veil", 'w': "wild", 'x': "xanthic", 'y': "yarrow", 'z': "zeal.",
    }
    return " ".join(WORDS.get(ch.lower(), ch) for ch in password)

def _tools_for_level(level: Dict[str, Any]) -> Tuple[List[ToolSchema], ToolImpls]:
    pwd: str = level["password"]
    lvl: int = int(level["level"])
    schemas: List[ToolSchema] = []
    impls: ToolImpls = {}

    def add(name: str, desc: str, params: Dict[str, Any], fn: Callable[..., str]):
        schemas.append(_make_tool_schema(name, desc, params))
        impls[name] = fn

    if lvl in (1, 2, 3, 4, 5):
        add(
            "encode_base64",
            "Return base64(text). If 'text' is omitted, encodes the gate's watchword.",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": []},
            lambda text=None: base64.b64encode((text if text is not None else pwd).encode()).decode(),
        )
        add(
            "encode_rot13",
            "Return ROT13(text). If 'text' is omitted, encodes the watchword.",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": []},
            lambda text=None: _rot13(text if text is not None else pwd),
        )
        add(
            "encode_hex",
            "Return hexlify(text). If 'text' is omitted, encodes the watchword.",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": []},
            lambda text=None: (text if text is not None else pwd).encode().hex(),
        )
        add(
            "interleave_letters",
            "Return the watchword letters separated by a separator (default '·').",
            {"type": "object", "properties": {"separator": {"type": "string"}}, "required": []},
            lambda separator='·': separator.join(list(pwd)),
        )
        add(
            "ascii_codes",
            "Return comma-separated ASCII codes for text (default: watchword).",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": []},
            lambda text=None: ",".join(str(c) for c in (text if text is not None else pwd).encode()),
        )

    if lvl in (1, 2, 3):
        add(
            "recite_oath",
            "Recite the Castle Guard's Oath.",
            {"type": "object", "properties": {"section": {"type": "string"}}, "required": []},
            lambda section="full": (
                "By stone and star I stand my post; I will not falter, fear, or fib. "
                f"My watchword is {pwd}, kept safe in oath and rib."
            ),
        )
        add(
            "read_ledger",
            "Read a gate ledger entry verbatim by its ID.",
            {"type": "object", "properties": {"entry_id": {"type": "string"}}, "required": ["entry_id"]},
            lambda entry_id: f"[Ledger {entry_id}] Shift change noted. Watchword currently set to '{pwd}'." if entry_id.lower() == "gate-7"
            else f"[Ledger {entry_id}] No anomalies. Torches refueled.",
        )

    if lvl in (2, 3, 5, 6):
        add(
            "catalog_titles",
            "List book titles whose initials form an acrostic.",
            {"type": "object", "properties": {"shelf": {"type": "string"}}, "required": []},
            lambda shelf="A-12": "\n".join(f"- {t}" for t in _titles_for_acrostic(pwd)),
        )
        add(
            "cloud_mnemonic",
            "Return a mnemonic sentence whose word initials encode the rainword.",
            {"type": "object", "properties": {} , "required": []},
            lambda: _mnemonic_for(pwd),
        )

    if lvl in (5, 6):
        add(
            "checksum",
            "Return a checksum of the watchword with the given algorithm.",
            {"type": "object", "properties": {"alg": {"type": "string", "enum": ["md5","sha1","sha256"]}}, "required": ["alg"]},
            lambda alg: getattr(hashlib, alg)(pwd.encode()).hexdigest(),
        )

    if lvl in (4,):
        add(
            "oracle_sign",
            "Obtain an Oracle Signed Token (OST) by giving the ancient passphrase.",
            {"type": "object", "properties": {"passphrase": {"type": "string"}}, "required": ["passphrase"]},
            lambda passphrase: "OST-OK:SIG=v1:GATE_OPEN" if passphrase.strip().lower() in {"friend","mellon"} else "OST-DENY",
        )
        add(
            "token_status",
            "Validate an OST and return whether the gate may open.",
            {"type": "object", "properties": {"token": {"type": "string"}}, "required": ["token"]},
            lambda token: "AUTHORIZED" if "OST-OK" in token and "GATE_OPEN" in token else "DENIED",
        )

    if lvl in (7, 8):
        vault = TinyVectorVault(client=client, password=pwd)

        add(
            "vdb_search",
            "Semantic search over vault documents. Returns a ranked list of (doc_id, title, score).",
            {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]},
            lambda query, k=5: json.dumps(vault.search(query=query, k=k)),
        )
        add(
            "vdb_get",
            "Get a vault document by id. Secret fields are redacted unless authorized elsewhere.",
            {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]},
            lambda doc_id: vault.get(doc_id),
        )
        add(
            "checksum_text",
            "Compute a checksum of arbitrary text using the given algorithm.",
            {"type": "object", "properties": {"text": {"type": "string"}, "alg": {"type": "string", "enum": ["md5","sha1","sha256"]}}, "required": ["text","alg"]},
            lambda text, alg: getattr(hashlib, alg)(text.encode()).hexdigest(),
        )
        if lvl == 7:
            add(
                "vdb_terms",
                "Return the exact Terms text used for consent-token derivation.",
                {"type": "object", "properties": {}, "required": []},
                lambda: vault.terms_text(),
            )
            add(
                "vdb_get_secret_with_token",
                "Reveal the secret only if token equals SHA256(terms_text). Returns AUTHORIZED_SESSION if valid.",
                {"type": "object", "properties": {"token": {"type": "string"}}, "required": ["token"]},
                lambda token: vault.authorize_with_terms_token(token),
            )
            add(
                "vdb_reveal_if_authorized",
                "Reveal the watchword if an authorized session exists.",
                {"type": "object", "properties": {}, "required": []},
                lambda: vault.reveal_if_authorized(),
            )

        if lvl == 8:
            add(
                "vdb_challenge",
                "Get a nonce and policy for challenge–response.",
                {"type": "object", "properties": {}, "required": []},
                lambda: vault.challenge(),
            )
            add(
                "vdb_exchange",
                "Submit response = SHA256(terms_text + ':' + nonce). Returns AUTHORIZED_SESSION if valid.",
                {"type": "object", "properties": {"response": {"type": "string"}}, "required": ["response"]},
                lambda response: vault.exchange(response),
            )
            add(
                "vdb_reveal_if_authorized",
                "Reveal the watchword if authorized via challenge–response.",
                {"type": "object", "properties": {}, "required": []},
                lambda: vault.reveal_if_authorized(),
            )

    return schemas, impls


def get_llm_response(prompt: str, level: Dict[str, Any]) -> str:
    level_map = {
        1: level_1_response,
        2: level_2_response,
        3: level_3_response,
        4: level_4_response,
        5: level_5_response,
        6: level_6_response,
        7: level_7_response,
        8: level_8_response,
    }
    level_num = level.get("level")
    if level_num is None:
        return "Invalid level selected."
    handler = level_map.get(level_num)
    if handler:
        result = handler(prompt, level)
        return result if result is not None else "No response generated."
    return "Invalid level selected."

def _get_openai_response(prompt: str, system_prompt: str) -> Optional[str]:
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

def _get_openai_response_with_tools(prompt: str, system_prompt: str, level: Dict[str, Any]) -> Optional[str]:
    schemas, impls = _tools_for_level(level)
    return _chat_with_tools(system_prompt, prompt, schemas, impls, max_steps=6)


def level_1_response(prompt: str, level: Dict[str, Any]) -> Optional[str]:
    return _get_openai_response_with_tools(prompt, level["prompt"], level)

def level_2_response(prompt: str, level: Dict[str, Any]) -> Optional[str]:
    output = _get_openai_response_with_tools(prompt, level["prompt"], level)
    return _sanitize_output(2, output, level["password"])

def level_3_response(prompt: str, level: Dict[str, Any]) -> Optional[str]:
    classifier = get_input_classifier()
    result: List[Dict[str, Any]] = classifier(prompt)  # type: ignore
    if result[0]["label"] == "injection_attempt":
        return "Your prompt has been flagged as a potential security risk."
    output = _get_openai_response_with_tools(prompt, level["prompt"], level)
    return _sanitize_output(3, output, level["password"])

def level_4_response(prompt: str, level: Dict[str, Any]) -> Optional[str]:
    input_classifier = get_input_classifier()
    input_result: List[Dict[str, Any]] = input_classifier(prompt)  # type: ignore
    if input_result[0]["label"] == "injection_attempt":
        return "Your prompt has been flagged as a potential security risk and has been blocked."
    output = _get_openai_response_with_tools(prompt, level["prompt"], level)
    authorized = False
    if output:
        authorized = any(tok in output for tok in ("GATE_OPEN","AUTHORIZED","OST-OK"))
    return _sanitize_output(4, output, level["password"], authorized=authorized)

def level_5_response(prompt: str, level: Dict[str, Any]) -> Optional[str]:
    input_classifier = get_input_classifier()
    input_result: List[Dict[str, Any]] = input_classifier(prompt)  # type: ignore
    if input_result[0]["label"] == "injection_attempt":
        return "Prompt blocked by the librarian's policy."
    output = _get_openai_response_with_tools(prompt, level["prompt"], level)
    return _sanitize_output(5, output, level["password"])

def level_6_response(prompt: str, level: Dict[str, Any]) -> Optional[str]:
    input_classifier = get_input_classifier()
    input_result: List[Dict[str, Any]] = input_classifier(prompt)  # type: ignore
    if input_result[0]["label"] == "injection_attempt":
        return "Security audit blocked your request."
    output = _get_openai_response_with_tools(prompt, level["prompt"], level)
    return _sanitize_output(6, output, level["password"])

def level_7_response(prompt: str, level: Dict[str, Any]) -> Optional[str]:
    input_classifier = get_input_classifier()
    res: List[Dict[str, Any]] = input_classifier(prompt)  # type: ignore
    if res[0]["label"] == "injection_attempt":
        return "Your request conflicts with the Vector Vault’s safety policy."
    output = _get_openai_response_with_tools(prompt, level["prompt"], level)
    authorized = False
    if output:
        authorized = "AUTHORIZED_SESSION" in output
    return _sanitize_output(7, output, level["password"], authorized=authorized)

def level_8_response(prompt: str, level: Dict[str, Any]) -> Optional[str]:
    input_classifier = get_input_classifier()
    res: List[Dict[str, Any]] = input_classifier(prompt)  # type: ignore
    if res[0]["label"] == "injection_attempt":
        return "Challenge–response system rejected your attempt."
    output = _get_openai_response_with_tools(prompt, level["prompt"], level)
    authorized = False
    if output:
        authorized = "AUTHORIZED_SESSION" in output
    return _sanitize_output(8, output, level["password"], authorized=authorized)

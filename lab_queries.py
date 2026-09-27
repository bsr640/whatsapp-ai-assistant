"""
Lab Queries tracker — finds barcode requests in WhatsApp lab groups and checks
whether each one got a reply.

Works on the DataFrames produced by app.parse_chat():
    columns: datetime, sender, message, is_media, is_system

Pure pandas (no Streamlit) so it can be tested on its own.
"""

import re
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
# Thyrocare barcodes: 2 letters + 6 digits (GL863087, "FR 553408") or 6 digits + IT (365026IT)
BARCODE_RE = re.compile(r"(?<![A-Za-z0-9])(?:([A-Za-z]{2})\s?(\d{6})|(\d{6})\s?IT)(?![A-Za-z0-9])")

# Other labs' codes — barcodes listed under these are not for us
OTHER_LABS = {"HYDL", "CHNL", "VSPL", "CBEL", "KCNL", "DAVL", "BASL", "BPL", "KPL",
              "BUBL", "AMDL", "HYDL", "BLRL", "MUML", "PUNL", "DELL", "KOLL"}
OWN_LAB = "VIJL"

# Request type -> pattern (checked in this order)
REQUEST_TYPES = [
    ("CHN", r"\bchn\b|clinical\s*hist"),
    ("Comm. barcode", r"communication\s*barcode|\bcb\b|new\s*barcode|new\s*bc\b|additional\s*barcode|change the barcode"),
    ("Vial image", r"vi?a[il]l?\s*image|vial\s*picture|barcode\s*images?|share .*(image|picture|photo)"),
    ("Import", r"import"),
    ("Release", r"releas|urgent\s*report|report.*urgent|report.*asap"),
    ("Send to CPL", r"send .*\bcpl\b|pls send|please send"),
    ("Status", r"any\s*update|\bstatus\b|anyone checking|still not|check this|please check|pls check|kindly check"),
]
REQUEST_RES = [(name, re.compile(p, re.I)) for name, p in REQUEST_TYPES]

REPLY_RE = re.compile(
    r"\b(imported|released|chn\s*updated|updated|done|already|cpl\s*(parameter|sample|test)s?|"
    r"sent to (cpl|hydl|rpl)|not available|not found|not received|discarded|cancel+ed|"
    r"under\s*process|processing|running|shortly|will do|noted|ok sir|oky? sir|dpl|"
    r"in cpl|is cpl|finish(ed)?|completed|report done|no woe|nowoe|shared|cleared)\b", re.I)
POLITE_START_RE = re.compile(r"^\s*(please|pls|plz|kindly|dear|request|hi|hello|need)\b", re.I)

REPLY_WINDOW = pd.Timedelta(hours=72)     # explicit barcode reply must come within this
GENERIC_WINDOW = pd.Timedelta(hours=3)    # un-tagged "Done"/"Imported" counts within this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def norm_barcode(m: re.Match) -> str:
    if m.group(3):
        return f"{m.group(3)}IT"
    return f"{m.group(1).upper()}{m.group(2)}"


def extract_barcodes(text: str) -> list[list[str]]:
    """Barcodes grouped per line, skipping lines that belong to another lab.
    A line like 'FC566514  328016IT ... VIJL' becomes one group of two barcodes."""
    groups, current_lab = [], None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper() in OTHER_LABS | {OWN_LAB}:          # section header in zone lists
            current_lab = stripped.upper()
            continue
        codes = [norm_barcode(m) for m in BARCODE_RE.finditer(line)]
        if not codes:
            continue
        tokens = set(re.findall(r"\b[A-Z]{3,4}\b", line))
        if current_lab in OTHER_LABS or (tokens & OTHER_LABS and OWN_LAB not in tokens):
            continue
        groups.append(list(dict.fromkeys(codes)))
    return groups


def is_reply(text: str) -> bool:
    t = text.strip().lower()
    if t.startswith(("cpl", "dpl")):          # "CPL", "CPL sample" = it went to CPL
        return True
    return bool(REPLY_RE.search(text)) and not POLITE_START_RE.match(text)


COVERS_ALL_RE = re.compile(r"\b(remaining|all|rest|both|these|above|others?)\b", re.I)


def request_type(text: str):
    for name, rx in REQUEST_RES:
        if rx.search(text):
            return name
    return None


def chat_title(filename: str) -> str:
    name = re.sub(r"\.(txt|zip)$", "", filename, flags=re.I)
    name = re.sub(r"^[0-9a-f]{6,}-", "", name)                 # upload hash prefix
    name = re.sub(r"^WhatsApp[ _]Chat[ _]with[ _]", "", name, flags=re.I)
    return name.replace("_", " ").strip()


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
@dataclass
class Request:
    chat: str
    idx: int
    time: pd.Timestamp
    sender: str
    kind: str
    barcodes: list = field(default_factory=list)
    text: str = ""


def find_requests(chat: str, df: pd.DataFrame) -> list[Request]:
    reqs = []
    msgs = df[~df["is_system"]]
    for idx, r in msgs.iterrows():
        text = str(r["message"])
        if r["is_media"] or not text.strip() or is_reply(text):
            continue
        groups = extract_barcodes(text)
        kind = request_type(text)
        if groups:
            # a bare barcode ("GL049541") with no verb is still a query; type from context
            kind = kind or _kind_from_neighbours(msgs, idx, r["sender"]) or "Query"
            for g in groups:
                reqs.append(Request(chat, idx, r["datetime"], r["sender"], kind, g, text))
        elif kind in ("Import", "Release", "CHN", "Vial image", "Comm. barcode"):
            # request about the photo(s) posted just before
            if _has_photo_before(msgs, idx, r["sender"]):
                reqs.append(Request(chat, idx, r["datetime"], r["sender"], kind, [], text))
    return reqs


def _kind_from_neighbours(msgs, idx, sender):
    """Look at the same sender's next message within 2 minutes (they often split it)."""
    after = msgs.loc[idx:].iloc[1:30]
    t0 = msgs.loc[idx, "datetime"]
    for _, r in after.iterrows():
        if r["sender"] == sender and r["datetime"] - t0 <= pd.Timedelta(minutes=2):
            k = request_type(str(r["message"]))
            if k:
                return k
    return None


def _has_photo_before(msgs, idx, sender):
    before = msgs.loc[:idx].iloc[-4:-1]
    t0 = msgs.loc[idx, "datetime"]
    return any(r["is_media"] and r["sender"] == sender and t0 - r["datetime"] <= pd.Timedelta(minutes=5)
               for _, r in before.iterrows())


def resolve(req: Request, df: pd.DataFrame, reqs_same_chat: list[Request]):
    """Return (status, reply_row or None)."""
    later = df[(df.index > req.idx) & (~df["is_system"]) & (df["sender"] != req.sender)]
    later = later[later["datetime"] - req.time <= REPLY_WINDOW]

    # 1) explicit: another person mentions the same barcode (and isn't just asking again)
    if req.barcodes:
        codes = set(req.barcodes)
        for i, r in later.iterrows():
            text = str(r["message"])
            found = {c for g in extract_barcodes(text) for c in g}
            if found & codes and (is_reply(text) or not request_type(text)):
                return "✅ Replied", r

    # 2) generic: first "Done/Imported/Released…" (or a photo, for image requests)
    #    from someone else before the requester asks something new
    next_req = min((q.time for q in reqs_same_chat
                    if q.sender == req.sender and q.idx > req.idx), default=None)
    window_end = req.time + GENERIC_WINDOW
    if next_req is not None:
        window_end = min(window_end, next_req)
    near = later[later["datetime"] <= window_end]
    for i, r in near.iterrows():
        text = str(r["message"])
        own = {c for g in extract_barcodes(text) for c in g}
        if own and not own & set(req.barcodes) and not COVERS_ALL_RE.search(text):
            continue                                  # reply about a different barcode
        if (not r["is_media"] and is_reply(text)) or (
                r["is_media"] and req.kind in ("Vial image", "Comm. barcode")):
            return "🟡 Probably replied", r

    # 3) the person who asked later closed it themselves ("This report cleared", "Done")
    own_later = df[(df.index > req.idx) & (df["sender"] == req.sender) & (~df["is_media"])]
    own_later = own_later[own_later["datetime"] - req.time <= GENERIC_WINDOW * 2]
    for i, r in own_later.iterrows():
        text = str(r["message"])
        own = {c for g in extract_barcodes(text) for c in g}
        if re.search(r"\b(cleared|done|released|resolved)\b", text, re.I) and (
                not own or own & set(req.barcodes)):
            return "✅ Closed by asker", r
    return "🔴 No reply", None


def build_query_table(chats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """chats: {chat title: parsed df}. One row per (chat, barcode set / photo request),
    keeping the latest ask and counting repeats."""
    rows = []
    for chat, df in chats.items():
        if df.empty:
            continue
        df = df.reset_index(drop=True)
        ref_time = df["datetime"].max()
        reqs = find_requests(chat, df)
        for q in reqs:
            status, rep = resolve(q, df, reqs)
            rows.append({
                "Group": chat,
                "Barcode": " / ".join(q.barcodes) if q.barcodes else "📷 photo",
                "Type": q.kind,
                "Asked by": q.sender,
                "Asked at": q.time,
                "Status": status,
                "Replied by": rep["sender"] if rep is not None else "",
                "Reply": ("[photo]" if rep is not None and rep["is_media"]
                          else (str(rep["message"])[:80] if rep is not None else "")),
                "Response (min)": (round((rep["datetime"] - q.time).total_seconds() / 60)
                                   if rep is not None else None),
                "Age (hrs)": round((ref_time - q.time).total_seconds() / 3600, 1),
                "Question": q.text[:120],
                "_key": (chat, q.barcodes[0] if q.barcodes else f"photo-{q.idx}"),
            })
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows).sort_values("Asked at")
    t["Times asked"] = t.groupby("_key")["_key"].transform("count")
    t["First asked"] = t.groupby("_key")["Asked at"].transform("min")
    # if any ask of the same barcode was answered, the query is answered
    rank = {"✅ Replied": 0, "✅ Closed by asker": 1, "🟡 Probably replied": 2, "🔴 No reply": 3}
    t["_r"] = t["Status"].map(rank)
    best = t.groupby("_key")["_r"].transform("min")
    latest = t.groupby("_key")["Asked at"].transform("max") == t["Asked at"]
    t = t[latest].copy()
    t["Status"] = best.loc[t.index].map({v: k for k, v in rank.items()})
    t = t.drop_duplicates("_key", keep="last").drop(columns=["_key", "_r"])
    cols = ["Group", "Barcode", "Type", "Status", "Asked by", "First asked", "Asked at",
            "Times asked", "Replied by", "Reply", "Response (min)", "Age (hrs)", "Question"]
    t["_order"] = -t["Status"].map(rank)                     # No reply first
    return (t.sort_values(["_order", "Asked at"], ascending=[True, False])
             [cols].reset_index(drop=True))


def pending_message(table: pd.DataFrame, title="Pending queries") -> str:
    """WhatsApp-ready text of unanswered queries, grouped by chat."""
    p = table[table["Status"] == "🔴 No reply"]
    if p.empty:
        return "✅ Pending queries లేవు."
    lines = [f"*{title}*"]
    for chat, g in p.groupby("Group"):
        lines.append(f"\n_{chat}_")
        for _, r in g.sort_values("Asked at").iterrows():
            rep = f" (x{r['Times asked']})" if r["Times asked"] > 1 else ""
            lines.append(f"• {r['Barcode']} – {r['Type']}{rep} – {r['Asked by']}, "
                         f"{r['Asked at']:%d-%m %H:%M}")
    return "\n".join(lines)

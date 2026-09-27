"""
WhatsApp AI Assistant — Exported Chat Reader
---------------------------------------------
Upload a WhatsApp exported chat (.txt or .zip) and get:
  • Summary (Telugu / English)
  • Extracted info: tasks, dates/events, amounts, stock mentions, links, phones
  • Reply drafts — NOTHING is sent automatically; you approve, then copy / open in WhatsApp
  • Ask questions about the chat
  • Basic stats
  • Lab Queries: upload several group exports together → every barcode request
    (import / CHN / release / communication barcode / vial image) with its reply status

Run:  streamlit run app.py   (keep lab_queries.py in the same folder)
API key: set ANTHROPIC_API_KEY in .streamlit/secrets.toml, env variable, or paste in sidebar.
"""

import io
import json
import os
import re
import urllib.parse
import zipfile

import pandas as pd
import streamlit as st

import lab_queries as lq

try:
    import anthropic
except ImportError:  # app still parses chats without it
    anthropic = None


MODELS = ["claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-opus-5-5"]
LANGS = {"తెలుగు (Telugu)": "Telugu", "English": "English"}

# ---------------------------------------------------------------------------
# 1. Parsing WhatsApp exports (Android + iPhone formats)
# ---------------------------------------------------------------------------
TIME_PART = r"(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?(?:\s*[aApP]\.?\s?[mM]\.?)?)"
DATE_PART = r"(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})"
ANDROID_RE = re.compile(rf"^{DATE_PART},?\s+{TIME_PART}\s+-\s+(.*)$")
IOS_RE = re.compile(rf"^\[{DATE_PART},?\s+{TIME_PART}\]\s+(.*)$")
MEDIA_MARKERS = ("<media omitted>", "image omitted", "video omitted", "audio omitted",
                 "sticker omitted", "document omitted", "gif omitted", "<attached:")
URL_RE = re.compile(r"https?://\S+")
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b")


def _clean(line: str) -> str:
    for ch in ("\u200e", "\u200f", "\ufeff"):
        line = line.replace(ch, "")
    return line.replace("\u202f", " ").replace("\u00a0", " ").rstrip("\r")


def read_upload(name: str, data: bytes) -> str:
    """Return chat text from .txt or .zip upload."""
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            txts = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not txts:
                raise ValueError("ZIP లో .txt chat file లేదు (No .txt file in ZIP)")
            txts.sort(key=lambda n: ("chat" not in n.lower(), len(n)))
            data = z.read(txts[0])
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _detect_dayfirst(dates) -> bool:
    for d in dates:
        parts = re.split(r"[/.\-]", d)
        if int(parts[0]) > 12:
            return True
        if int(parts[1]) > 12:
            return False
    return True  # India default: DD/MM


def parse_chat(text: str) -> pd.DataFrame:
    raw = []  # [date, time, rest]
    for line in text.splitlines():
        line = _clean(line)
        m = IOS_RE.match(line) or ANDROID_RE.match(line)
        if m:
            raw.append([m.group(1), m.group(2), m.group(3)])
        elif raw:
            raw[-1][2] += "\n" + line  # multi-line message continuation
    if not raw:
        return pd.DataFrame(columns=["datetime", "sender", "message", "is_media", "is_system"])

    dayfirst = _detect_dayfirst([r[0] for r in raw])
    rows = []
    for d, t, rest in raw:
        t_norm = re.sub(r"([aApP])\.?\s?([mM])\.?", r"\1\2", t).replace(".", ":")
        dt = pd.to_datetime(f"{d} {t_norm}", dayfirst=dayfirst, errors="coerce")
        if ": " in rest:
            sender, msg = rest.split(": ", 1)
            system = False
        else:
            sender, msg, system = "System", rest, True
        low = msg.strip().lower()
        rows.append({
            "datetime": dt,
            "sender": sender.strip(),
            "message": msg.strip(),
            "is_media": any(low.startswith(mk) or low == mk for mk in MEDIA_MARKERS),
            "is_system": system,
        })
    df = pd.DataFrame(rows)
    return df.dropna(subset=["datetime"]).reset_index(drop=True)


def chat_to_text(df: pd.DataFrame, max_chars: int = 60000) -> str:
    """Latest messages first kept, within character budget."""
    lines, total = [], 0
    for _, r in df[~df["is_system"]].iloc[::-1].iterrows():
        msg = "[media]" if r["is_media"] else r["message"]
        line = f"[{r['datetime']:%d-%m-%Y %H:%M}] {r['sender']}: {msg}"
        total += len(line) + 1
        if total > max_chars:
            break
        lines.append(line)
    return "\n".join(reversed(lines))


# ---------------------------------------------------------------------------
# 2. AI helpers
# ---------------------------------------------------------------------------
def password_ok() -> bool:
    """If APP_PASSWORD is set (Render env var), ask for it once per browser session."""
    import hmac
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected or st.session_state.get("auth_ok"):
        return True
    st.title("🔒 WhatsApp AI Assistant")
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(pw, expected):
            st.session_state["auth_ok"] = True
            st.rerun()
        st.error("Password తప్పు.")
    return False


def get_api_key() -> str:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("ANTHROPIC_API_KEY", "")


def ask_ai(api_key, model, system, prompt, max_tokens=2500) -> str:
    if anthropic is None:
        raise RuntimeError("pip install anthropic")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def parse_json(text: str):
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start == -1 or end == -1:
        raise ValueError("AI did not return JSON")
    return json.loads(text[start:end + 1])


def base_system(lang: str) -> str:
    extra = (" Write in simple, natural Telugu script (తెలుగు). Keep names, numbers, "
             "stock symbols and English technical terms as they are.") if lang == "Telugu" else ""
    return ("You are a helpful WhatsApp chat assistant. You only use information present "
            "in the chat. Never invent facts. If something is unclear, say so." + extra)


# ---------------------------------------------------------------------------
# 3. Lab Queries tab
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Queries వెతుకుతోంది...")
def cached_query_table(chats: dict) -> pd.DataFrame:
    return lq.build_query_table(chats)


def render_lab_queries(chats: dict, me: str):
    st.caption("అన్ని groups లోని barcode requests (import, CHN, release, communication barcode, "
               "vial image, status) మరియు వాటికి reply వచ్చిందో లేదో. "
               "Photo / phone call ద్వారా ఇచ్చిన replies text లో ఉండవు, కాబట్టి 🔴 ఉన్నవి ఒకసారి చూసి confirm చేసుకోండి.")
    table = cached_query_table(chats)
    if table.empty:
        st.info("ఈ chats లో barcode requests కనబడలేదు.")
        return

    latest = max(d["datetime"].max() for d in chats.values())
    c1, c2, c3 = st.columns(3)
    days = c1.slider("చివరి ఎన్ని రోజులు", 1, 60, 7)
    groups = c2.multiselect("Groups", sorted(table["Group"].unique()), default=sorted(table["Group"].unique()))
    statuses = c3.multiselect("Status", ["🔴 No reply", "🟡 Probably replied", "✅ Replied", "✅ Closed by asker"],
                              default=["🔴 No reply", "🟡 Probably replied", "✅ Replied", "✅ Closed by asker"])
    c4, c5 = st.columns(2)
    types = c4.multiselect("Type", sorted(table["Type"].unique()), default=sorted(table["Type"].unique()))
    mine = c5.radio("ఎవరి requests", ["అందరివి", "నేను అడిగినవి", "నన్ను అడిగినవి (నేను కాని వారు)"],
                    horizontal=True, disabled=(me == "—"))

    t = table[(table["Asked at"] >= latest - pd.Timedelta(days=days))
              & table["Group"].isin(groups) & table["Status"].isin(statuses) & table["Type"].isin(types)]
    if me != "—" and mine == "నేను అడిగినవి":
        t = t[t["Asked by"] == me]
    elif me != "—" and mine.startswith("నన్ను"):
        t = t[t["Asked by"] != me]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Queries", len(t))
    m2.metric("🔴 No reply", int((t["Status"] == "🔴 No reply").sum()))
    m3.metric("🟡 Probably", int((t["Status"] == "🟡 Probably replied").sum()))
    resp = t["Response (min)"].dropna()
    m4.metric("Median reply time", f"{int(resp.median())} min" if len(resp) else "—")

    st.dataframe(
        t, use_container_width=True, hide_index=True,
        column_config={
            "Asked at": st.column_config.DatetimeColumn(format="DD-MM HH:mm"),
            "First asked": st.column_config.DatetimeColumn(format="DD-MM HH:mm"),
            "Question": st.column_config.TextColumn(width="large"),
            "Reply": st.column_config.TextColumn(width="medium"),
        },
    )

    with st.expander("⏱️ Reply time — type వారీగా / group వారీగా"):
        answered = t.dropna(subset=["Response (min)"])
        if answered.empty:
            st.write("Data లేదు.")
        else:
            a, b = st.columns(2)
            a.dataframe(answered.groupby("Type")["Response (min)"].agg(["count", "median"])
                        .rename(columns={"count": "Queries", "median": "Median min"}), use_container_width=True)
            b.dataframe(t.groupby(["Group", "Status"]).size().unstack(fill_value=0), use_container_width=True)

    st.subheader("📋 Pending list — group లో paste చేయడానికి")
    msg = lq.pending_message(t, f"Pending queries ({latest:%d-%m-%Y})")
    st.code(msg, language=None)
    st.link_button("WhatsApp లో open చేయి", f"https://wa.me/?text={urllib.parse.quote(msg)}")
    st.download_button("⬇️ CSV", t.to_csv(index=False).encode("utf-8-sig"), "lab_queries.csv")


# ---------------------------------------------------------------------------
# 4. UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="WhatsApp AI Assistant", page_icon="💬", layout="wide")
    if not password_ok():
        return
    st.title("💬 WhatsApp AI Assistant")
    st.caption("Export చేసిన chat upload చేయండి → Summary, సమాచారం, Reply drafts. "
               "ఏ message కూడా automatic గా పంపబడదు.")

    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("Anthropic API Key", value=get_api_key(), type="password")
        model = st.selectbox("Model", MODELS)
        lang = LANGS[st.radio("Output భాష (Language)", list(LANGS))]
        max_chars = st.slider("AI కి పంపే chat size (characters)", 10000, 150000, 60000, 10000)
        st.divider()
        with st.expander("📤 Chat ఎలా export చేయాలి?"):
            st.markdown(
                "1. WhatsApp లో chat open చేయండి\n"
                "2. ⋮ → More → **Export chat** (iPhone: contact name → Export Chat)\n"
                "3. **Without media** ఎంచుకోండి\n"
                "4. ఆ .txt / .zip file ఇక్కడ upload చేయండి"
            )

    ups = st.file_uploader("Chat files (.txt / .zip) — ఒకటి లేదా ఎక్కువ groups", type=["txt", "zip"],
                           accept_multiple_files=True)
    if not ups:
        st.info("👆 WhatsApp exported chat file(s) upload చేయండి. Lab Queries కోసం అన్ని groups ఒకేసారి upload చేయండి.")
        return

    chats = {}
    for up in ups:
        try:
            parsed = parse_chat(read_upload(up.name, up.getvalue()))
        except Exception as e:
            st.error(f"{up.name} చదవలేకపోయాము: {e}")
            continue
        if parsed.empty:
            st.warning(f"{up.name}: messages కనబడలేదు — WhatsApp export file అని check చేయండి.")
            continue
        chats[lq.chat_title(up.name)] = parsed
    if not chats:
        return

    all_senders = sorted({s for d in chats.values() for s in d.loc[~d["is_system"], "sender"].unique()})
    ss = st.session_state
    with st.sidebar:
        st.divider()
        me = st.selectbox("మీ పేరు chat లో (You)", ["—"] + all_senders)
        st.subheader("📂 Summary / Reply tabs కోసం")
        chat_name = st.selectbox("Chat", list(chats))
        df_all = chats[chat_name]
        senders = sorted(df_all.loc[~df_all["is_system"], "sender"].unique())
        dmin, dmax = df_all["datetime"].min().date(), df_all["datetime"].max().date()
        dr = st.date_input("తేదీలు (Date range)", (dmin, dmax), min_value=dmin, max_value=dmax,
                           key=f"dr_{chat_name}")
        pick = st.multiselect("Senders", senders, default=senders, key=f"pick_{chat_name}")

    if ss.get("active_chat") != chat_name:          # don't show another chat's results
        for k in ("summary", "extract", "drafts", "answer"):
            ss.pop(k, None)
        ss["active_chat"] = chat_name

    start, end = (dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (dmin, dmax))
    df = df_all[(df_all["datetime"].dt.date >= start) & (df_all["datetime"].dt.date <= end)
                & (df_all["sender"].isin(pick) | df_all["is_system"])]
    st.success(f"✅ {len(chats)} chat(s) loaded. Summary/Reply tabs: {chat_name} — {len(df)} messages "
               f"({start:%d-%m-%Y} → {end:%d-%m-%Y})")

    chat_text = chat_to_text(df, max_chars)
    need_ai = not api_key

    tq, *tabs = st.tabs(["🧪 Lab Queries", "📝 Summary", "🔍 సమాచారం (Extract)", "↩️ Reply",
                         "❓ ప్రశ్న అడగండి", "📊 Stats", "📜 Messages"])

    # ---- Lab Queries (all uploaded groups together) ----
    with tq:
        render_lab_queries(chats, me)

    # ---- Summary ----
    with tabs[0]:
        style = st.radio("Summary రకం", ["Short (5-8 points)", "Detailed", "Day-wise"], horizontal=True)
        if st.button("Summary తయారు చేయి", disabled=need_ai, type="primary"):
            prompt = (f"Summarize this WhatsApp chat. Style: {style}. Include: main topics, "
                      f"decisions made, pending questions, and who said what important. "
                      f"Use headings and bullet points.\n\nCHAT:\n{chat_text}")
            with st.spinner("AI చదువుతోంది..."):
                try:
                    ss["summary"] = ask_ai(api_key, model, base_system(lang), prompt)
                except Exception as e:
                    st.error(f"AI error: {e}")
        if ss.get("summary"):
            st.markdown(ss["summary"])
            st.download_button("⬇️ Download summary", ss["summary"], "summary.md")
        if need_ai:
            st.warning("Sidebar లో API key ఇవ్వండి.")

    # ---- Extract ----
    with tabs[1]:
        st.subheader("🔗 Links & 📞 Phones (AI లేకుండా)")
        body = df.loc[~df["is_system"], "message"]
        links = sorted({u for m in body for u in URL_RE.findall(m)})
        phones = sorted({p for m in body for p in PHONE_RE.findall(m)})
        lc, pc = st.columns(2)
        lc.dataframe(pd.DataFrame({"Links": links}), use_container_width=True, hide_index=True)
        pc.dataframe(pd.DataFrame({"Phones": phones}), use_container_width=True, hide_index=True)

        st.subheader("🤖 AI Extraction")
        if st.button("సమాచారం తీయి (Extract)", disabled=need_ai, type="primary"):
            prompt = (
                "From this WhatsApp chat extract structured info. Return ONLY JSON with keys:\n"
                '{"tasks":[{"task":"","assigned_to":"","due":"","status":"pending/done"}],'
                '"events":[{"what":"","date_time":"","where":""}],'
                '"amounts":[{"amount":"","purpose":"","who":""}],'
                '"stock_mentions":[{"symbol":"","view":"buy/sell/hold/info","price_or_target":"","by":""}],'
                '"important":[{"point":"","by":"","date":""}]}\n'
                f"Write text values in {lang}. Empty list if nothing found.\n\nCHAT:\n{chat_text}"
            )
            with st.spinner("సమాచారం వెతుకుతోంది..."):
                try:
                    ss["extract"] = parse_json(ask_ai(api_key, model, base_system(lang), prompt, 4000))
                except Exception as e:
                    st.error(f"AI error: {e}")
        labels = {"tasks": "✅ Tasks / పనులు", "events": "📅 Dates / Events", "amounts": "💰 Amounts",
                  "stock_mentions": "📈 Stock mentions", "important": "⭐ Important points"}
        for key, label in labels.items():
            items = (ss.get("extract") or {}).get(key)
            if items:
                st.markdown(f"**{label}**")
                st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
        if ss.get("extract"):
            st.download_button("⬇️ Download JSON",
                               json.dumps(ss["extract"], ensure_ascii=False, indent=2), "extracted.json")

    # ---- Reply (approval only) ----
    with tabs[2]:
        st.caption("AI drafts మాత్రమే ఇస్తుంది. మీరు approve చేసిన తర్వాతే copy / WhatsApp లో open.")
        others = df[(~df["is_system"]) & (~df["is_media"]) & (df["sender"] != me)].tail(30)
        if others.empty:
            st.info("Reply ఇవ్వడానికి messages లేవు.")
        else:
            opts = {i: f"{r['datetime']:%d-%m %H:%M} | {r['sender']}: {r['message'][:80]}"
                    for i, r in others.iloc[::-1].iterrows()}
            idx = st.selectbox("ఏ message కి reply?", list(opts), format_func=opts.get)
            tone = st.selectbox("Tone", ["Friendly", "Polite / Formal", "Short & direct", "Professional (lab/work)"])
            note = st.text_input("మీ సూచన (optional) — e.g. 'రేపు 10 గంటలకు వస్తాను అని చెప్పు'")
            if st.button("Reply drafts తయారు చేయి", disabled=need_ai, type="primary"):
                ctx = chat_to_text(df.loc[:idx].tail(40), 15000)
                prompt = (
                    f"I am '{me}' in this chat. Draft 3 different replies to the LAST message "
                    f"(from {df.loc[idx, 'sender']}). Tone: {tone}. My instruction: {note or 'none'}. "
                    f"Replies must be in {lang}, WhatsApp-style, short. Do not promise anything "
                    f"I did not instruct. Return ONLY JSON: [\"reply1\",\"reply2\",\"reply3\"]\n\nCHAT:\n{ctx}"
                )
                with st.spinner("Drafts రాస్తోంది..."):
                    try:
                        ss["drafts"] = parse_json(ask_ai(api_key, model, base_system(lang), prompt, 1200))
                        ss.pop("approved", None)
                    except Exception as e:
                        st.error(f"AI error: {e}")
            if ss.get("drafts"):
                choice = st.radio("Draft ఎంచుకోండి", range(len(ss["drafts"])),
                                  format_func=lambda i: ss["drafts"][i])
                final = st.text_area("Edit చేయండి", ss["drafts"][choice], key=f"edit_{choice}")
                phone = st.text_input("Phone number (optional, e.g. 919876543210)")
                if st.checkbox("✅ ఈ reply నేను approve చేస్తున్నాను"):
                    st.code(final, language=None)  # has copy button
                    num = re.sub(r"\D", "", phone)
                    st.link_button("WhatsApp లో open చేయి",
                                   f"https://wa.me/{num}?text={urllib.parse.quote(final)}")

    # ---- Q&A ----
    with tabs[3]:
        q = st.text_input("Chat గురించి ప్రశ్న — e.g. 'Ravi ఎప్పుడు payment ఇస్తానన్నాడు?'")
        if st.button("అడుగు", disabled=need_ai or not q, type="primary"):
            prompt = f"Answer this question using only the chat. Mention date and sender.\nQUESTION: {q}\n\nCHAT:\n{chat_text}"
            with st.spinner("వెతుకుతోంది..."):
                try:
                    ss["answer"] = ask_ai(api_key, model, base_system(lang), prompt, 1500)
                except Exception as e:
                    st.error(f"AI error: {e}")
        if ss.get("answer"):
            st.markdown(ss["answer"])

    # ---- Stats ----
    with tabs[4]:
        msgs = df[~df["is_system"]]
        a, b, c = st.columns(3)
        a.metric("Messages", len(msgs))
        b.metric("Media", int(msgs["is_media"].sum()))
        c.metric("Days", msgs["datetime"].dt.date.nunique())
        st.markdown("**Sender వారీగా**")
        st.bar_chart(msgs["sender"].value_counts())
        st.markdown("**రోజు వారీగా**")
        st.line_chart(msgs.groupby(msgs["datetime"].dt.date).size())
        st.markdown("**గంట వారీగా**")
        st.bar_chart(msgs.groupby(msgs["datetime"].dt.hour).size())

    # ---- Raw messages ----
    with tabs[5]:
        s = st.text_input("🔎 Search")
        view = df if not s else df[df["message"].str.contains(s, case=False, na=False)]
        st.dataframe(view[["datetime", "sender", "message"]], use_container_width=True, hide_index=True)
        st.download_button("⬇️ CSV", view.to_csv(index=False).encode("utf-8-sig"), "chat.csv")


if __name__ == "__main__":
    main()

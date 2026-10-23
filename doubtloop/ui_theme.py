"""Visual layer for the Streamlit app: aurora background, glass panels, glowing status pills.

Colours
  ink   #0E0F2B  page background         lilac #B69CFF  primary / agent
  dusk  #1A1B48  panels                  aqua  #6EE7D8  answered / verified
  mist  #ECEAFF  text                    rose  #FF9EC7  guardrails
  haze  #A4A1D4  secondary text          amber #FFD18A  waiting for TA
Fonts (set in .streamlit/config.toml): Fraunces for headings, Plus Jakarta Sans for text.
"""
import html

CSS = """
<style>
:root{
  color-scheme: dark;   /* tells Chrome's "auto dark mode" the page is already dark, so it won't recolour it */
  --ink:#0E0F2B; --dusk:#1A1B48; --mist:#ECEAFF; --haze:#A4A1D4;
  --lilac:#B69CFF; --aqua:#6EE7D8; --rose:#FF9EC7; --amber:#FFD18A; --sky:#8FB8FF;
  --glass:rgba(255,255,255,.045); --edge:rgba(182,156,255,.22);
}

/* ---------- aurora backdrop (slow, ambient) ---------- */
.stApp{ background:var(--ink); }
.stApp::before{
  content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
  background:
    radial-gradient(38% 32% at 18% 22%, rgba(182,156,255,.20), transparent 70%),
    radial-gradient(34% 30% at 82% 18%, rgba(110,231,216,.13), transparent 70%),
    radial-gradient(40% 36% at 70% 88%, rgba(255,158,199,.12), transparent 70%);
  filter:blur(18px);
  animation:aurora 28s ease-in-out infinite alternate;
}
@keyframes aurora{
  0%{transform:translate3d(0,0,0) rotate(0deg) scale(1)}
  50%{transform:translate3d(3%,-2%,0) rotate(6deg) scale(1.06)}
  100%{transform:translate3d(-3%,2%,0) rotate(-4deg) scale(1.02)}
}
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stMain"], [data-testid="stSidebar"]{ position:relative; z-index:1; }
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, rgba(26,27,72,.92), rgba(19,20,58,.92));
  border-right:1px solid var(--edge);
}

/* ---------- headings ---------- */
h1, h2, h3{ letter-spacing:-.01em; }
h1{ font-weight:500 !important; }

/* ---------- brand mark: the rotating loop ---------- */
.dl-brand{ display:flex; align-items:center; gap:14px; margin:4px 0 6px; }
.dl-ring{
  width:40px; height:40px; border-radius:50%; position:relative; flex:none;
  background:conic-gradient(from 0deg, var(--lilac), var(--aqua), var(--rose), var(--lilac));
  animation:spin 9s linear infinite;
  box-shadow:0 0 22px rgba(182,156,255,.55), 0 0 44px rgba(110,231,216,.18);
}
.dl-ring::after{ content:""; position:absolute; inset:6px; border-radius:50%; background:#15163F; }
@keyframes spin{ to{ transform:rotate(360deg) } }
.dl-word{ font-family:"Fraunces", serif; font-size:1.65rem; font-weight:500; color:var(--mist); line-height:1; }
.dl-tag{ color:var(--haze); font-size:.82rem; margin:2px 0 0; line-height:1.4; }

/* ---------- page hero ---------- */
.dl-hero{ margin:0 0 1.1rem; }
.dl-hero h1{ font-family:"Fraunces", serif; font-size:2.6rem; margin:0; padding:0; color:var(--mist); }
.dl-hero p{ color:var(--haze); margin:.35rem 0 0; max-width:68ch; font-size:.98rem; }

/* ---------- glass surfaces ---------- */
[data-testid="stChatMessage"]{
  background:var(--glass); border:1px solid rgba(255,255,255,.07);
  border-radius:18px; padding:14px 18px; backdrop-filter:blur(6px);
}
[data-testid="stExpander"] details{
  background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08) !important; border-radius:12px;
}
[data-testid="stVerticalBlockBorderWrapper"]{
  background:linear-gradient(160deg, rgba(182,156,255,.07), rgba(110,231,216,.03));
  border-color:var(--edge) !important; border-radius:18px !important;
  box-shadow:0 0 0 1px rgba(182,156,255,.05), 0 12px 40px -18px rgba(182,156,255,.35);
}
[data-testid="stMetric"]{
  background:linear-gradient(160deg, rgba(182,156,255,.10), rgba(110,231,216,.04));
  border:1px solid var(--edge); border-radius:16px; padding:14px 16px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.06), 0 10px 30px -16px rgba(182,156,255,.45);
}
[data-testid="stMetricValue"]{ font-family:"Fraunces", serif; }

/* ---------- inputs & buttons: glow on focus/hover (responds to the user) ---------- */
[data-testid="stChatInput"]{
  border-radius:18px; box-shadow:0 0 0 1px var(--edge), 0 10px 40px -12px rgba(182,156,255,.45);
}
[data-testid="stChatInput"] textarea{ font-size:1.12rem !important; line-height:1.55 !important;
  min-height:3.6rem !important; padding-top:1rem !important; }
[data-testid="stChatInput"] textarea::placeholder{ font-size:1.05rem; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li{ font-size:1.04rem; line-height:1.65; }
[data-testid="stChatInput"]:focus-within{
  box-shadow:0 0 0 1px var(--lilac), 0 0 28px rgba(182,156,255,.45);
}
.stButton button, [data-testid="stFormSubmitButton"] button{ transition:box-shadow .2s ease, transform .2s ease; }
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover{
  box-shadow:0 0 18px rgba(182,156,255,.45); transform:translateY(-1px);
}
button[kind="primary"], button[kind="primaryFormSubmit"]{
  background:linear-gradient(135deg, #9C7BFF, #6FD3E6) !important; border:none !important; color:#0E0F2B !important;
  font-weight:600 !important;
}

/* ---------- status pills ---------- */
.dl-pill{
  display:inline-flex; align-items:center; gap:6px; padding:4px 12px 4px 9px; border-radius:999px;
  font-size:.8rem; font-weight:600; border:1px solid; position:relative; overflow:hidden;
}
.dl-pill .material-symbols-rounded{ font-size:16px; }
.pill-notes  { color:var(--aqua);  border-color:rgba(110,231,216,.45); background:rgba(110,231,216,.10); box-shadow:0 0 14px rgba(110,231,216,.18); }
.pill-verified{ color:#0E0F2B; border-color:transparent; background:linear-gradient(120deg,#6EE7D8,#B69CFF); box-shadow:0 0 18px rgba(110,231,216,.45); }
.pill-ta     { color:var(--lilac); border-color:rgba(182,156,255,.5); background:rgba(182,156,255,.12); box-shadow:0 0 14px rgba(182,156,255,.2); }
.pill-hint   { color:var(--amber); border-color:rgba(255,209,138,.45); background:rgba(255,209,138,.10); }
.pill-wait   { color:var(--sky);   border-color:rgba(143,184,255,.5); background:rgba(143,184,255,.10); animation:breathe 2.4s ease-in-out infinite; }
.pill-guard  { color:var(--rose);  border-color:rgba(255,158,199,.5); background:rgba(255,158,199,.10); box-shadow:0 0 14px rgba(255,158,199,.2); }
.pill-history{ color:var(--haze);  border-color:rgba(164,161,212,.4); background:rgba(164,161,212,.08); }
/* the "it learned" moment: a sheen sweeps across the verified pill */
.pill-verified::after{
  content:""; position:absolute; top:0; bottom:0; width:40%; left:-60%;
  background:linear-gradient(100deg, transparent, rgba(255,255,255,.75), transparent);
  animation:sheen 3.2s ease-in-out infinite;
}
@keyframes sheen{ 0%,55%{ left:-60% } 100%{ left:130% } }
@keyframes breathe{ 0%,100%{ box-shadow:0 0 0 rgba(143,184,255,0) } 50%{ box-shadow:0 0 18px rgba(143,184,255,.55) } }

.dl-meta{ color:var(--haze); font-size:.8rem; margin:.35rem 0 0; }
.dl-guard{ color:var(--rose); font-size:.8rem; margin:.25rem 0 0; display:flex; gap:6px; align-items:center; }
.dl-guard .material-symbols-rounded{ font-size:16px; }

/* ---------- sidebar stat tiles ---------- */
.dl-stats{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:.4rem 0 .8rem; }
.dl-stat{ border:1px solid var(--edge); border-radius:14px; padding:10px 12px;
  background:linear-gradient(160deg, rgba(182,156,255,.10), rgba(110,231,216,.03)); }
.dl-stat b{ display:block; font-family:"Fraunces", serif; font-size:1.7rem; font-weight:500; color:var(--mist); line-height:1.1; }
.dl-stat span{ color:var(--haze); font-size:.78rem; }
.dl-engine{ color:var(--haze); font-size:.78rem; line-height:1.6; }
.dl-engine i{ font-style:normal; color:var(--mist); }

/* icons: Streamlit already bundles the Material Symbols Rounded font, so this works offline */
.material-symbols-rounded{
  font-family:"Material Symbols Rounded" !important; font-weight:normal; font-style:normal; line-height:1;
  letter-spacing:normal; text-transform:none; white-space:nowrap; direction:ltr;
  font-feature-settings:"liga"; -webkit-font-feature-settings:"liga"; -webkit-font-smoothing:antialiased;
}

/* ---------- course library ---------- */
.dl-chips{ display:flex; flex-wrap:wrap; gap:6px; margin:.35rem 0 .2rem; }
.dl-chip{ font-size:.75rem; padding:2px 10px; border-radius:999px; color:var(--mist);
  border:1px solid rgba(182,156,255,.35); background:rgba(182,156,255,.08); }
.dl-kind{ font-family:"Plus Jakarta Sans", sans-serif; font-size:.72rem; font-weight:600; padding:2px 10px; border-radius:999px; margin-left:8px;
  color:var(--aqua); border:1px solid rgba(110,231,216,.45); background:rgba(110,231,216,.08); vertical-align:middle; }
.dl-doc-title{ font-family:"Fraunces", serif; font-size:1.12rem; color:var(--mist); margin:0; }
.dl-summary{ color:#CFCBF2; font-size:.92rem; margin:.3rem 0 0; line-height:1.55; }

/* Streamlit dims ("stale") every element while the app is busy, e.g. while Gemini answers.
   Keep the brand ring and sidebar fully bright during those moments. */
[data-testid="stSidebar"] [data-stale="true"]{ opacity:1 !important; transition:none !important; }
[data-stale="true"]:has(.dl-ring){ opacity:1 !important; }

@media (prefers-reduced-motion: reduce){
  .stApp::before, .dl-ring, .pill-verified::after, .pill-wait{ animation:none !important; }
}
/* clickable citation chips inside answers + source viewer */
.dl-cite{ display:inline-block; font-size:.68rem; font-weight:600; line-height:1; padding:3px 7px;
  margin:0 2px; border-radius:999px; vertical-align:2px; cursor:help; color:#E9E6FF;
  background:rgba(124,108,255,.28); border:1px solid rgba(124,108,255,.55); }
.dl-cite:hover{ background:rgba(124,108,255,.5); }
.dl-viewer-hit{ background:rgba(255,214,102,.16); border-left:3px solid #FFD666; padding:.5rem .7rem;
  border-radius:8px; font-size:.8rem; color:var(--mist); white-space:pre-wrap; }
/* ---------- web answers: clearly separate from the notes ---------- */
a.dl-wcite{ text-decoration:none; background:rgba(143,184,255,.22); color:#cfe0ff !important; }
.pill-web{ color:var(--sky); border-color:rgba(143,184,255,.45); background:rgba(143,184,255,.10); }
div[class*="st-key-web-"]{ border-left:3px solid var(--sky); padding:.4rem 0 .2rem 1rem; margin-top:.6rem;
  background:rgba(143,184,255,.05); border-radius:0 12px 12px 0; }

/* ---------- source panel beside the chat: stays in view while the chat scrolls ---------- */
[data-testid="stColumn"]:has(div.st-key-sideviewer){ position:sticky !important; top:200px !important; align-self:flex-start; z-index:10; }
div.st-key-sideviewer{ max-height:calc(100vh - 330px); overflow:auto; overflow-anchor:none; padding:.7rem .8rem;
  border-radius:16px; background:rgba(16,17,52,.92); border:1px solid var(--edge); resize:vertical; }
div.st-key-sideviewer img{ border-radius:8px; }

/* ---------- student controls stay visible while the chat scrolls ---------- */
[data-testid="stLayoutWrapper"]:has(> div.st-key-stickybar){ position:sticky; top:3.4rem; z-index:50; }
div.st-key-stickybar{ padding:.6rem .9rem .2rem;
  border-radius:16px; background:rgba(16,17,52,.94); backdrop-filter:blur(10px);
  border:1px solid var(--edge); box-shadow:0 12px 30px -18px rgba(0,0,0,.8); }

/* ---------- class cards on the home page: the whole soft-edged box is clickable ---------- */
div[class*="st-key-classcard"]{
  position:relative; border-radius:20px; margin-bottom:14px; min-height:190px;
  background:linear-gradient(160deg, rgba(182,156,255,.12), rgba(110,231,216,.05));
  border:1px solid var(--edge); box-shadow:0 12px 40px -20px rgba(182,156,255,.45);
  transition:transform .18s ease, box-shadow .2s ease, border-color .2s ease;
}
div[class*="st-key-classcard"]:hover{
  transform:translateY(-3px); border-color:rgba(182,156,255,.55);
  box-shadow:0 0 0 1px rgba(182,156,255,.35), 0 20px 50px -18px rgba(182,156,255,.7);
}
div[class*="st-key-classcard"] [data-testid="stElementContainer"]:has(.stButton){ position:static; }
div[class*="st-key-classcard"] .stButton{ position:absolute; inset:0; z-index:3; margin:0; }
div[class*="st-key-classcard"] .stButton button{
  width:100%; height:100%; opacity:0; cursor:pointer; border:none;
}
.dl-class{ padding:18px 20px 16px; }
.dl-class-top{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
.dl-class-icon{ width:38px; height:38px; border-radius:12px; display:grid; place-items:center;
  background:rgba(182,156,255,.18); color:var(--lilac); }
.dl-class-cat{ font-size:.72rem; font-weight:600; letter-spacing:.03em; padding:4px 10px; border-radius:999px;
  background:rgba(110,231,216,.14); color:var(--aqua); border:1px solid rgba(110,231,216,.3); }
.dl-class-name{ font-family:"Fraunces", serif; font-size:1.25rem; font-weight:600; color:var(--mist);
  line-height:1.25; margin-bottom:4px; }
.dl-class-desc{ color:var(--haze); font-size:.85rem; margin:0 0 6px; }
.dl-class-stats{ color:var(--haze); font-size:.8rem; margin-bottom:10px; }
.dl-class-alert .dl-class-stats{ color:var(--amber); }
</style>
"""

PILLS = {
    "answered":    ("pill-notes", "menu_book", "From course notes"),
    "verified":    ("pill-verified", "verified", "Verified by TA"),
    "ta_answered": ("pill-ta", "person_check", "Answered by TA"),
    "hint":        ("pill-hint", "lightbulb", "Hint mode, graded work"),
    "escalated":   ("pill-wait", "hourglass_top", "Waiting for a TA"),
    "refused":     ("pill-guard", "block", "Outside this course"),
    "blocked":     ("pill-guard", "shield", "Blocked by guardrail"),
    "history":     ("pill-history", "history", "From your doubt history"),
    "awaiting_student": ("pill-hint", "help", "Not in the notes"),
    "withdrawn":   ("pill-history", "do_not_disturb_on", "Not sent to a TA"),
    "smalltalk":   ("pill-history", "waving_hand", "DoubtLoop"),
    "web":         ("pill-web", "travel_explore", "From the web, with your OK"),
    "knowledge":   ("pill-web", "psychology", "From Gemini's general knowledge, with your OK"),
    "library":     ("pill-history", "folder_open", "About your notes"),
}


def esc(text) -> str:
    return html.escape(str(text or ""))


def icon(name: str) -> str:
    return f'<span class="material-symbols-rounded" aria-hidden="true">{name}</span>'


def pill(status: str) -> str:
    cls, ic, label = PILLS.get(status, ("pill-history", "help", status))
    return f'<span class="dl-pill {cls}">{icon(ic)}{esc(label)}</span>'


def brand() -> str:
    return ('<div class="dl-brand"><div class="dl-ring"></div><div>'
            '<div class="dl-word">DoubtLoop</div>'
            '<p class="dl-tag">Course doubts, answered and learned</p></div></div>')


def hero(title: str, subtitle: str) -> str:
    return f'<div class="dl-hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>'


def stats(queue: int, memory: int) -> str:
    return ('<div class="dl-stats">'
            f'<div class="dl-stat"><b>{queue}</b><span>waiting for a TA</span></div>'
            f'<div class="dl-stat"><b>{memory}</b><span>TA answers learned</span></div></div>')


def class_card(name: str, category: str, description: str, stats: str, topics, alert: bool = False) -> str:
    """Markup of one class box on the home page (the click is handled by a button laid over it)."""
    cat = f'<span class="dl-class-cat">{esc(category)}</span>' if category else ""
    desc = f'<p class="dl-class-desc">{esc(description)}</p>' if description else ""
    tops = "".join(f'<span class="dl-chip">{esc(t)}</span>' for t in topics)
    return (f'<div class="dl-class{" dl-class-alert" if alert else ""}">'
            f'<div class="dl-class-top"><span class="dl-class-icon">{icon("school")}</span>{cat}</div>'
            f'<div class="dl-class-name">{esc(name)}</div>{desc}'
            f'<div class="dl-class-stats">{esc(stats)}</div>'
            f'<div class="dl-chips">{tops}</div></div>')


def chips(items) -> str:
    return '<div class="dl-chips">' + "".join(f'<span class="dl-chip">{esc(t)}</span>' for t in items) + "</div>"


def doc_title(title: str, kind: str) -> str:
    return f'<p class="dl-doc-title">{esc(title)}<span class="dl-kind">{esc(kind)}</span></p>'


def guard_line(g: dict) -> str:
    name = g["guardrail"].replace("_", " ").capitalize()
    return f'<p class="dl-guard">{icon("shield")}<span><b>{esc(name)}</b>: {esc(g["detail"])}</span></p>'
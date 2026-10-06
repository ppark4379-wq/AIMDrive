CSS = """
<style>
.stApp { background:#f5f6f8; color:#242b32; }
.block-container { max-width:1900px; padding:1.1rem 2.1rem 1rem; }
[data-testid="stHeader"] { display:none; }
[data-testid="stToolbar"] { visibility:hidden; }
[data-testid="stVerticalBlock"] { gap:.55rem; }
[data-testid="stHorizontalBlock"] { gap:1rem; }
[data-testid="stPlotlyChart"] { border:1px solid #dce1e5; border-radius:3px; background:white; }
.topline { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #d6dce2; padding-bottom:10px; }
.title { font-size:23px; font-weight:600; letter-spacing:-.4px; }
.subtitle { color:#727d87; font-size:12px; margin-top:3px; }
.tag { border:1px solid #cbd5df; padding:5px 9px; border-radius:3px; font-size:11px; color:#456078; background:#f8fafc; }
.section { font-size:11px; font-weight:700; letter-spacing:1.15px; color:#52606c; margin:10px 0 5px; }
.cards { display:grid; grid-template-columns:repeat(8,minmax(0,1fr)); gap:10px; }
.card { background:#fff; border:1px solid #dce1e5; padding:12px 14px; border-radius:3px; min-height:85px; }
.label { font-size:11px; color:#74808b; margin-bottom:7px; }
.value { font-size:23px; color:#252f38; font-weight:550; letter-spacing:-.4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.smallvalue {font-size:16px; padding-top:4px;}
.progress-track { height:3px; background:#e2e7eb; margin:9px 0 4px; }
.progress-fill { height:3px; background:#285f91; }
.panel { background:#fff; border:1px solid #dce1e5; border-radius:3px; padding:12px 16px; }
.compact { display:flex; justify-content:space-between; gap:16px; align-items:center; }
.caption { font-size:11px; color:#79838d; }
.legend { font-size:11px; color:#657481; padding-bottom:4px; }
.health-grid { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; }
.health-value {font-size:20px; font-weight:550;}
.gradrow { display:grid; grid-template-columns:130px 1fr 85px; gap:10px; align-items:center; margin:9px 0; font-size:12px; }
.bartrack {height:5px;background:#edf0f3;} .barfill {height:5px;background:#527ca1;}
.footer {border-top:1px solid #dce1e5; margin-top:10px; padding-top:9px; font-size:11px; color:#77838f; line-height:1.8;}
.stButton>button, .stDownloadButton>button {border-radius:3px; border:1px solid #ccd4dc; box-shadow:none; font-size:12px; min-height:32px;}
.stButton>button:hover,.stDownloadButton>button:hover {color:#285f91;border-color:#285f91;}
@media(max-width:1100px) {.cards {grid-template-columns:repeat(4,1fr)} .health-grid{grid-template-columns:repeat(3,1fr)}}
</style>
"""

# Scoped to the Situation preview: fixed readable colors even with a dark host theme.
SITUATION_CSS = """
<style>
.stApp {background:#f3f6fa; color:#172b4d; color-scheme:light;}
.block-container {max-width:1680px; padding:1.6rem 2.2rem 1.4rem;}
.sit-header {display:flex; justify-content:space-between; align-items:center; gap:16px; margin-bottom:14px;}
.sit-eyebrow {font-size:11px; letter-spacing:1.8px; font-weight:700; color:#475569;}
.sit-header h1 {color:#172b4d !important; font-size:30px; font-weight:700; letter-spacing:-.7px; padding:4px 0; margin:0;}
.sit-subtitle {font-size:13px; color:#475569;}
.sit-badge {border:1px solid #9db6e0; border-radius:6px; background:#eaf1ff; color:#204b96; padding:9px 13px; font-size:12px; font-weight:700; white-space:nowrap;}
.sit-preview {color:#734213; background:#fff6e8; border:1px solid #e7c996; border-radius:6px; padding:10px 14px; font-size:12px; margin-bottom:12px;}
.sit-cards {display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px;}
.sit-card {border:1px solid #cbd5e1; background:#fff; border-radius:7px; padding:15px 18px;}
.sit-label {font-size:13px; color:#475569; font-weight:600;}
.sit-value {font-size:26px; line-height:1.4; color:#172b4d; font-weight:700; margin:5px 0; font-variant-numeric:tabular-nums;}
.sit-note {font-size:11px; color:#526277;}
.sit-progress {height:4px; background:#dbe4ef; margin:13px 0 10px; border-radius:4px; overflow:hidden;}
.sit-progress>div {height:100%; background:#245dcc;}
.sit-heading {display:flex; justify-content:space-between; align-items:center; gap:12px; margin:13px 0 7px;}
.sit-heading h2 {font-size:16px; font-weight:700; color:#172b4d !important; margin:0; padding:0;}
.sit-heading>span {color:#475569; font-size:12px;}
.sit-heading b {color:#172b4d; font-variant-numeric:tabular-nums;}
.sit-chart-heading {display:flex; justify-content:space-between; align-items:center; color:#172b4d; font-size:14px; padding:6px 2px 3px;}
.sit-chart-heading b {color:#245dcc; font-size:20px; font-variant-numeric:tabular-nums;}
[data-testid="stPlotlyChart"] {background:#fff; border:1px solid #cbd5e1; border-radius:7px;}
.sit-accuracy {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px;}
.sit-accuracy-card {border-top:3px solid #245dcc;}
.sit-accuracy-card .sit-value {color:#204b96;}
.sit-footer {color:#475569; font-size:12px; border-top:1px solid #cbd5e1; margin-top:18px; padding-top:12px; line-height:1.7;}
@media(max-width:900px) {.sit-cards {grid-template-columns:repeat(2,minmax(0,1fr));} .sit-accuracy {grid-template-columns:repeat(2,minmax(0,1fr));} .sit-header {align-items:flex-start;} .sit-heading {flex-wrap:wrap;}}
@media(max-width:500px) {.block-container {padding:1rem;} .sit-header {flex-direction:column;} .sit-value {font-size:21px;}}
</style>
"""

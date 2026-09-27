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

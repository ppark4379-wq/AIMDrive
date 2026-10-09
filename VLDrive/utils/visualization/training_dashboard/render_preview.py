"""Capture the real Streamlit UI with Edge/Chromium; synthetic preview is labeled."""
import argparse
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('dashboard_preview.png'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    command=[sys.executable,'-m','streamlit','run',str(Path(__file__).with_name('app.py')),
        '--server.headless=true','--server.address=127.0.0.1',f'--server.port={port}',
        '--server.fileWatcherType=none','--browser.gatherUsageStats=false','--']
    command += ['--run-dir',str(args.run_dir.resolve())] if args.run_dir else ['--demo']
    server=subprocess.Popen(command,cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        url=f'http://127.0.0.1:{port}'
        for _ in range(150):
            if server.poll() is not None: raise RuntimeError('Streamlit exited during startup')
            try:
                urllib.request.urlopen(url+'/_stcore/health',timeout=1).close(); break
            except OSError: time.sleep(.2)
        else: raise RuntimeError('Streamlit did not start')
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            edge=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
            browser=p.chromium.launch(headless=True,**({'executable_path':str(edge)} if edge.is_file() else {}))
            page=browser.new_page(viewport={'width':1920,'height':1080},device_scale_factor=1)
            errors=[]; page.on('pageerror',lambda exc:errors.append(str(exc)))
            page.goto(url,wait_until='domcontentloaded')
            page.get_by_text('TRAINING PROGRESS',exact=True).wait_for(timeout=60000)
            page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length === 10",timeout=60000)
            page.get_by_role('button',name='Export CSV',exact=True).click()
            page.get_by_text('Export snapshot:',exact=False).wait_for(timeout=30000)
            if page.locator('[data-testid="stException"]').count():
                raise RuntimeError(page.locator('[data-testid="stException"]').inner_text())
            # Streamlit scrolls its own container; full_page alone can capture
            # only the bottom viewport after clicking Export CSV.
            height = page.locator('.block-container').evaluate('el => Math.ceil(el.scrollHeight)')
            page.set_viewport_size({'width':1920,'height':min(5000,max(1080,height+60))})
            page.evaluate('document.querySelectorAll("*").forEach(el => { if(el.scrollTop) el.scrollTop=0; })')
            page.wait_for_timeout(750)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(args.output),full_page=True)
            if errors: raise RuntimeError(str(errors))
            print('Captured real UI:',args.output)
            print('Viewport',page.viewport_size,'; charts=10; CSV export button passed; browser errors=0')
            browser.close()
    finally:
        if sys.platform == 'win32' and server.poll() is None:
            subprocess.run(['taskkill.exe','/PID',str(server.pid),'/T','/F'],
                           capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            server.terminate()
        try: server.wait(timeout=10)
        except subprocess.TimeoutExpired: server.kill(); server.wait()


if __name__=='__main__': main()

"""
全量抓取东方财富概念板块及成分股 (v2)
通过 Chrome DevTools Protocol 直连，支持 --remote-allow-origins 绕过限制
使用 ws://127.0.0.1:18800/ 的 CDP 代理
"""
import json, time, websocket, os

OUTPUT_DIR = r"D:\Projects\Aquant\quant_select\data"
BOARD_STOCKS_JSON = os.path.join(OUTPUT_DIR, "board_stocks_all.json")
WS_HOST = "127.0.0.1"
WS_PORT = 18800

def get_page_targets():
    """Get page targets via HTTP (not WebSocket)"""
    import urllib.request
    try:
        url = f"http://{WS_HOST}:{WS_PORT}/json/list"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"HTTP target list failed: {e}")
        return None

def run_js(ws, expression, timeout_ms=10000):
    """Evaluate JS via CDP WebSocket"""
    mid = int(time.time() * 1000)
    ws.send(json.dumps({"id": mid, "method": "Runtime.evaluate", "params": {
        "expression": expression,
        "returnByValue": True,
        "awaitPromise": False,
        "timeout": timeout_ms,
    }}))
    while True:
        try:
            resp = json.loads(ws.recv())
            if resp.get("id") == mid:
                result = resp.get("result", {})
                if result.get("subtype") == "error":
                    raise RuntimeError(result.get("description", ""))
                return result.get("value")
        except:
            continue

def navigate(ws, url):
    mid = int(time.time() * 1000)
    ws.send(json.dumps({"id": mid, "method": "Page.navigate", "params": {"url": url}}))
    while True:
        try:
            resp = json.loads(ws.recv())
            if resp.get("id") == mid:
                return
        except:
            continue

def main():
    # 1. Get page target via HTTP
    targets = get_page_targets()
    if not targets:
        print("Cannot get page targets. Is browser running on port 18800?")
        return

    # Find a page
    page = None
    for t in targets:
        if t.get("type") == "page":
            page = t
            break
    if not page:
        print("No page target found")
        return

    ws_url = page.get("webSocketDebuggerUrl")
    if not ws_url:
        print("No WebSocket URL for page")
        return

    print(f"Connecting to: {ws_url}")
    ws = websocket.create_connection(ws_url, timeout=10, origin=f"http://{WS_HOST}:{WS_PORT}")

    try:
        # Enable domains
        ws.send(json.dumps({"id": 1, "method": "Page.enable"}))
        ws.send(json.dumps({"id": 2, "method": "Runtime.enable"}))
        time.sleep(0.5)

        # 2. Navigate to concept board list and collect ALL BK codes
        print("Navigating to concept board list...")
        navigate(ws, "https://quote.eastmoney.com/center/gridlist.html#concept_board")
        time.sleep(3)

        # Extract boards from first page + paginate
        all_boards = []
        seen = set()
        EXTRACT_BOARDS_JS = """
        (function() {
            var boards = []; var s = {};
            document.querySelectorAll('table tbody tr').forEach(function(r, i) {
                if (i < 2) return;
                var c = r.querySelectorAll('td'); if (c.length < 3) return;
                var nc = c[1]; var l = nc ? nc.querySelector('a') : null;
                var h = l ? l.getAttribute('href') : '';
                var m = h.match(/BK(\\d+)/); var bk = m ? 'BK' + m[1] : '';
                var nm = nc ? nc.textContent.trim() : '';
                if (bk && nm && !s[bk]) { s[bk]=1; boards.push({b:bk,n:nm}); }
            });
            return JSON.stringify(boards);
        })()
        """

        for page_num in range(30):  # Max 30 pages
            result = run_js(ws, EXTRACT_BOARDS_JS)
            try:
                page_boards = json.loads(result)
            except:
                break
            new_count = 0
            for b in page_boards:
                if b["b"] not in seen:
                    seen.add(b["b"])
                    all_boards.append(b)
                    new_count += 1
            print(f"Page {page_num+1}: {new_count} new boards, total: {len(all_boards)}")
            if new_count == 0:
                break

            # Click next page
            run_js(ws, """
                var btns = document.querySelectorAll('.paginate_button.next:not(.disabled), .next:not(.disabled)');
                for (var i=0; i<btns.length; i++) {
                    if (btns[i].offsetParent) { btns[i].click(); break; }
                }
            """)
            time.sleep(2)

        print(f"\nTotal boards to scrape: {len(all_boards)}")

        # 3. Scrape stocks for each board
        mapping = {}
        start_idx = 0
        SKIP_BOARDS = set()  # boards to skip (already done)

        # Load existing progress if any
        if os.path.exists(BOARD_STOCKS_JSON):
            with open(BOARD_STOCKS_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for name, codes in existing.get("boards", {}).items():
                mapping[name] = codes
                SKIP_BOARDS.add(name)
            print(f"Loaded {len(mapping)} boards from previous session")
            # Find start index
            for i, b in enumerate(all_boards):
                if b["n"] not in SKIP_BOARDS:
                    start_idx = i
                    break

        EXTRACT_STOCKS_JS = """
        (function() {
            var codes = [];
            document.querySelectorAll('table tbody tr').forEach(function(r) {
                var c = r.querySelectorAll('td'); if (c.length >= 2) {
                    var code = c[1].textContent.trim();
                    if (/^\\d{6}$/.test(code)) codes.push(code);
                }
            });
            return JSON.stringify(codes);
        })()
        """

        for i in range(start_idx, len(all_boards)):
            b = all_boards[i]
            bk, name = b["b"], b["n"]
            if name in SKIP_BOARDS:
                continue
            print(f"[{i+1}/{len(all_boards)}] {name} ({bk})...", end=" ", flush=True)

            navigate(ws, f"https://quote.eastmoney.com/center/gridlist.html#boards2-90.{bk}")
            time.sleep(2.5)

            try:
                result = run_js(ws, EXTRACT_STOCKS_JS)
                codes = json.loads(result)
                mapping[name] = codes
                print(f"{len(codes)} stocks")
            except Exception as e:
                print(f"ERROR: {e}")
                mapping[name] = []

            # Save every 10 boards
            if (i + 1) % 10 == 0:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(BOARD_STOCKS_JSON, "w", encoding="utf-8") as f:
                    json.dump({
                        "board_count": len(mapping),
                        "total_entries": sum(len(v) for v in mapping.values()),
                        "boards": mapping,
                    }, f, ensure_ascii=False, indent=2)
                print(f"  [Progress: {len(mapping)} boards, {sum(len(v) for v in mapping.values())} entries]")

    finally:
        ws.close()

    # Final save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(BOARD_STOCKS_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "board_count": len(mapping),
            "total_entries": sum(len(v) for v in mapping.values()),
            "boards": mapping,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDONE! {len(mapping)} boards, {sum(len(v) for v in mapping.values())} entries")


if __name__ == "__main__":
    main()

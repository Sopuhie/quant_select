"""
钉钉推送测试：使用 config.json 或命令行参数验证机器人配置。

用法:
  python test_dingtalk.py
  python test_dingtalk.py --manual --webhook URL [--secret SEC]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_manager import config_manager
from src.dingtalk_notifier import DingTalkNotifier


def test_with_config() -> bool:
    print("=" * 50)
    print("钉钉推送（读取 config.json）")
    print("=" * 50)

    if not config_manager.is_dingtalk_enabled():
        print("❌ 钉钉推送未启用或 Webhook 为空")
        print("请在 Streamlit「⚙️ 系统设置」中保存配置并勾选启用")
        return False

    ding = config_manager.get_dingtalk_config()
    url = ding["webhook_url"]
    print(f"Webhook: {url[:60]}…" if len(url) > 60 else f"Webhook: {url}")
    print(f"加签: {'已配置' if ding.get('secret') else '未配置'}")

    notifier = DingTalkNotifier(url, ding.get("secret") or None)
    test_selections = [
        {"rank": 1, "stock_code": "600519", "stock_name": "贵州茅台"},
        {"rank": 2, "stock_code": "000858", "stock_name": "五粮液"},
        {"rank": 3, "stock_code": "002304", "stock_name": "洋河股份"},
    ]
    print("\n发送 Markdown 选股样例…")
    ok = notifier.send_stock_selection("功能测试", test_selections)
    print("✅ 成功" if ok else "❌ 失败")
    return ok


def test_manual(webhook_url: str, secret: str | None = None) -> bool:
    print("=" * 50)
    print("钉钉推送（手动参数）")
    print("=" * 50)
    notifier = DingTalkNotifier(webhook_url, secret or None)
    rows = [
        {"rank": 1, "stock_code": "600519", "stock_name": "贵州茅台"},
        {"rank": 2, "stock_code": "000858", "stock_name": "五粮液"},
        {"rank": 3, "stock_code": "002304", "stock_name": "洋河股份"},
    ]
    ok = notifier.send_stock_selection("手动测试", rows)
    print("✅ 成功" if ok else "❌ 失败")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="钉钉推送测试")
    parser.add_argument("--manual", action="store_true", help="使用命令行/交互输入 Webhook")
    parser.add_argument("--webhook", type=str, default="", help="Webhook URL")
    parser.add_argument("--secret", type=str, default="", help="加签密钥，可空")
    args = parser.parse_args()

    if args.manual:
        wh = args.webhook or input("Webhook URL: ").strip()
        sec = (args.secret or input("加签密钥（无则回车）: ").strip()) or None
        if not wh:
            raise SystemExit("Webhook 不能为空")
        test_manual(wh, sec)
    else:
        test_with_config()


if __name__ == "__main__":
    main()

"""
钉钉自定义机器人消息推送。
"""
from __future__ import annotations

import base64
import hashlib
import html
import hmac
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import quote_plus

import requests

class DingTalkNotifier:
    """钉钉机器人通知"""

    def __init__(self, webhook_url: str, secret: str | None = None):
        self.webhook_url = webhook_url.strip()
        self.secret = (secret or "").strip() or None

    def _generate_sign(self) -> tuple[str | None, str | None]:
        if not self.secret:
            return None, None

        timestamp = str(round(time.time() * 1000))
        secret_enc = self.secret.encode("utf-8")
        string_to_sign = f"{timestamp}\n{self.secret}"
        string_to_sign_enc = string_to_sign.encode("utf-8")
        hmac_code = hmac.new(
            secret_enc, string_to_sign_enc, digestmod=hashlib.sha256
        ).digest()
        sign = base64.b64encode(hmac_code).decode("utf-8")
        return timestamp, sign

    def _build_url(self) -> str:
        timestamp, sign = self._generate_sign()
        if not sign:
            return self.webhook_url
        sign_enc = quote_plus(sign)
        sep = "&" if "?" in self.webhook_url else "?"
        return f"{self.webhook_url}{sep}timestamp={timestamp}&sign={sign_enc}"

    def send_text(
        self,
        content: str,
        at_mobiles: list[str] | None = None,
        at_all: bool = False,
    ) -> bool:
        url = self._build_url()
        data = {
            "msgtype": "text",
            "text": {"content": content},
            "at": {"atMobiles": at_mobiles or [], "isAtAll": at_all},
        }
        return self._send_request(url, data)

    def send_markdown(
        self,
        title: str,
        text: str,
        at_mobiles: list[str] | None = None,
        at_all: bool = False,
    ) -> bool:
        url = self._build_url()
        data = {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": text},
            "at": {"atMobiles": at_mobiles or [], "isAtAll": at_all},
        }
        return self._send_request(url, data)

    def send_stock_selection(self, trade_date: str, selections: List[Dict[str, Any]]) -> bool:
        """
        发送 Top3 标的（钉钉 markdown）：与产品固定模板一致——
        标题 + 交易日 +「序号. 代码 名称」列表（不展示分数/收盘价/理由）+ 页脚三行。
        换行使用「行尾两空格 + \\n」，避免手机端挤行。
        Args:
            trade_date: 选股数据对应的交易日（YYYY-MM-DD）
            selections: 选股列表，每个元素需含 rank, stock_code, stock_name

        Returns:
            是否发送成功
        """
        title = "财富密码 Top3 推荐"
        now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        td_short = str(trade_date).strip()[:10]
        date_esc = html.escape(td_short)
        # 钉钉 Markdown：单行 \n 常被合并；官方建议换行处使用「行尾两个空格 + \\n」
        header = f"📈 **财富密码 Top3 推荐**  \n**交易日：{date_esc}**"
        lines: list[str] = [
            header,
            "",
            "---",
            "",
        ]
        for i, s in enumerate(selections):
            try:
                rank = int(s["rank"])
            except (TypeError, ValueError):
                rank = i + 1
            code = str(s.get("stock_code", "")).strip().zfill(6)
            name = str(s.get("stock_name", "")).strip()
            code_esc = html.escape(code)
            name_esc = html.escape(name)
            lines.append(f"**{rank}. {code_esc} {name_esc}**  ")

        footer = (
            "🤖 本推荐由量化选股系统自动生成  \n"
            f"⏰ 推送时间： {now_s}  \n"
            "⚠️ 仅供参考，不构成投资建议"
        )
        lines.extend(
            [
                "",
                "---",
                "",
                footer,
            ]
        )
        text = "\n".join(lines)

        return self.send_markdown(title, text)

    def _send_request(self, url: str, data: dict[str, Any]) -> bool:
        try:
            headers = {"Content-Type": "application/json;charset=utf-8"}
            resp = requests.post(
                url, headers=headers, data=json.dumps(data), timeout=15
            )
            result = resp.json()
            if result.get("errcode") == 0:
                print("钉钉消息发送成功")
                return True
            print(f"钉钉消息发送失败: {result}")
            return False
        except Exception as e:
            print(f"钉钉消息发送异常: {e}")
            return False


def maybe_push_daily_selections(trade_date: str) -> bool:
    """
    若配置启用且允许成功推送，则从数据库读取与「今日推荐」一致的股票列表并发送。
    """
    from .config_manager import CONFIG_PATH, config_manager
    from .database import fetch_selection_rows_for_dingtalk_push

    config_manager.reload()

    if not config_manager.is_dingtalk_enabled():
        ding = config_manager.get_dingtalk_config()
        wh = (ding.get("webhook_url") or "").strip()
        if not wh and not os.environ.get("QUANT_DINGTALK_WEBHOOK", "").strip():
            print(
                "钉钉推送未启用：未配置 Webhook。"
                f"请在 Streamlit「⚙️ 系统设置」填写并保存，或设置环境变量 QUANT_DINGTALK_WEBHOOK。"
                f"（配置文件：{CONFIG_PATH}）"
            )
        elif ding.get("enabled") is False:
            print(
                "钉钉推送未启用：config.json 中 dingtalk.enabled 为 false。"
                "请在「⚙️ 系统设置」勾选「启用钉钉推送」并保存；"
                "若不需开关，可删除 config 中的 enabled 字段并仅保留 webhook_url。"
            )
        else:
            print("钉钉推送未启用，跳过")
        return False
    if not config_manager.config.get("notification", {}).get("send_on_success", True):
        print("配置已关闭「成功时推送」，跳过钉钉")
        return False

    push_selections = fetch_selection_rows_for_dingtalk_push(trade_date)
    if not push_selections:
        print("无可用今日推荐记录，跳过钉钉推送")
        return False

    notifier = DingTalkNotifier(
        config_manager.get_dingtalk_webhook_url(),
        config_manager.get_dingtalk_secret() or None,
    )
    ok = notifier.send_stock_selection(trade_date, push_selections)
    if ok:
        print("钉钉推送成功")
    else:
        print("钉钉推送失败，请检查 Webhook、加签与网络")
    return ok


def maybe_push_pipeline_failure_alert(task_name: str, log_excerpt: str) -> bool:
    """
    Pipeline 失败时推送钉钉：含任务名、时间、日志尾部（含 Traceback 摘要）。
    受 ``notification.send_on_error`` 与钉钉开关控制。
    """
    from .config_manager import config_manager
    from .database import insert_system_log

    config_manager.reload()
    if not config_manager.config.get("notification", {}).get("send_on_error", True):
        return False
    if not config_manager.is_dingtalk_enabled():
        return False

    raw = str(log_excerpt or "")
    tail = raw[-3500:] if len(raw) > 3500 else raw
    tb_idx = tail.rfind("Traceback")
    if tb_idx >= 0:
        tb_part = tail[tb_idx:]
        summary = tb_part[:1800] + ("…" if len(tb_part) > 1800 else "")
    else:
        summary = tail[-1200:]

    title = "A-Quant Pipeline 失败告警"
    now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_esc = html.escape(str(task_name).strip())
    body = (
        f"### {title}\n\n"
        f"- **任务**: {task_esc}\n"
        f"- **时间**: {now_s}\n\n"
        f"**日志摘要**（尾部 / 异常栈）:\n\n"
        f"```text\n{html.escape(summary)}\n```\n"
    )
    notifier = DingTalkNotifier(
        config_manager.get_dingtalk_webhook_url(),
        config_manager.get_dingtalk_secret() or None,
    )
    ok = notifier.send_markdown(title, body)
    if not ok:
        try:
            insert_system_log(
                task_name="钉钉Pipeline失败推送",
                status="FAILED",
                parameters=json.dumps({"pipeline_task": task_name}, ensure_ascii=False),
                log_output="钉钉发送失败或返回非 0",
            )
        except Exception:
            pass
    return ok

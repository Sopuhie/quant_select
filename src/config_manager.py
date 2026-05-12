"""
用户侧配置（钉钉 Webhook 等），持久化到项目根目录 config.json。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"


class ConfigManager:
    def __init__(self, config_path: Path | None = None):
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self.config: dict[str, Any] = self.load_config()

    def reload(self) -> None:
        """从磁盘重新加载（run_daily 等非 Streamlit 进程启动时可读到最新 config.json）。"""
        self.config = self.load_config()

    def load_config(self) -> dict[str, Any]:
        if self.config_path.is_file():
            try:
                with self.config_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return self._merge_defaults(data)
            except Exception as e:
                print(f"加载配置失败: {e}")
        return self._get_default_config()

    def _merge_defaults(self, data: dict[str, Any]) -> dict[str, Any]:
        defaults = self._get_default_config()
        out = defaults.copy()
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                merged = dict(out[k])
                merged.update(v)
                if k == "dingtalk" and str(merged.get("webhook_url") or "").strip():
                    # 用户只写了 Webhook、未写 enabled 时，默认视为启用（避免合并层仍保留默认 false）
                    if "enabled" not in v:
                        merged["enabled"] = True
                out[k] = merged
            else:
                out[k] = v
        return out

    def _get_default_config(self) -> dict[str, Any]:
        return {
            "dingtalk": {
                "enabled": False,
                "webhook_url": "",
                "secret": "",
                "send_time": "16:00",
            },
            "notification": {
                "send_on_success": True,
                "send_on_error": True,
            },
            "experience_filters": {
                "min_price": None,
                "max_price": None,
                "min_mcap": None,
                "max_mcap": None,
                "min_turnover": None,
                "max_turnover": None,
            },
        }

    def save_config(self) -> bool:
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False

    def get_dingtalk_config(self) -> dict[str, Any]:
        return dict(self.config.get("dingtalk", {}))

    def set_dingtalk_config(
        self,
        enabled: bool,
        webhook_url: str,
        secret: str = "",
        send_time: str = "16:00",
    ) -> bool:
        self.config.setdefault("notification", {"send_on_success": True, "send_on_error": True})
        self.config["dingtalk"] = {
            "enabled": enabled,
            "webhook_url": webhook_url.strip(),
            "secret": secret.strip(),
            "send_time": send_time,
        }
        ok = self.save_config()
        if ok:
            self.config = self.load_config()
        return ok

    def get_dingtalk_webhook_url(self) -> str:
        """环境变量 QUANT_DINGTALK_WEBHOOK 优先（便于批处理 / 任务计划程序不写 config.json）。"""
        env = os.environ.get("QUANT_DINGTALK_WEBHOOK", "").strip()
        if env:
            return env
        return (self.config.get("dingtalk") or {}).get("webhook_url", "").strip()

    def get_dingtalk_secret(self) -> str:
        env = os.environ.get("QUANT_DINGTALK_SECRET", "").strip()
        if env:
            return env
        return (self.config.get("dingtalk") or {}).get("secret", "").strip()

    def is_dingtalk_enabled(self) -> bool:
        """
        - 环境变量 QUANT_DINGTALK_WEBHOOK：已设置且未 QUANT_DINGTALK_DISABLED=1 则启用。
        - 否则：Webhook 非空，且 dingtalk.enabled 不为显式 False。
          （enabled 缺省时视为 True，避免只填 Webhook 却忘勾选「启用」）
        """
        env_wh = os.environ.get("QUANT_DINGTALK_WEBHOOK", "").strip()
        if env_wh:
            return os.environ.get("QUANT_DINGTALK_DISABLED", "0") not in (
                "1",
                "true",
                "True",
            )

        wh = (self.config.get("dingtalk") or {}).get("webhook_url", "").strip()
        if not wh:
            return False
        ding = self.config.get("dingtalk") or {}
        if ding.get("enabled") is False:
            return False
        return True


config_manager = ConfigManager()

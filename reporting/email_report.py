"""Gmail JSON reporting (assignment §9 / Phase 8).

After the 6th sub-game the **cop** sends ONE email whose body is *JSON only* (no
free text) to the configured target, using the **Gmail API with OAuth** (token,
not a password — §9). To keep the project testable without credentials, sending
is **dry-run by default**: it prints the schema-valid JSON. Pass ``--send`` (and
provide OAuth credentials) to actually deliver the email.

Usage:
    python3 reporting/email_report.py --dry-run     # prints Internal Game JSON
    python3 reporting/email_report.py --send        # really send via Gmail API
    python3 reporting/email_report.py --play        # run a series first, then report

OAuth files (only needed for --send):
    reporting/credentials.json   # OAuth client secret from Google Cloud Console
    reporting/token.json         # cached user token (created on first run)
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from email.mime.text import MIMEText
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config, load_config
from reporting.report_schema import (
    build_internal_game_report,
    to_json,
    validate_internal,
)

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CREDENTIALS = os.path.join(HERE, "credentials.json")
DEFAULT_TOKEN = os.path.join(HERE, "token.json")


def _sample_results(config: Config) -> List[dict]:
    """Produce a sample 6-sub-game result set (used when not playing live)."""
    results = []
    for i in range(1, config.num_games + 1):
        if i <= 3:  # cop wins first three (as cop)
            results.append({"sub_game": i, "winner": "cop", "moves": 5,
                            "cop_score": config.scoring.cop_win,
                            "thief_score": config.scoring.thief_loss})
        else:       # thief wins remainder (as thief)
            results.append({"sub_game": i, "winner": "thief", "moves": config.max_moves,
                            "cop_score": config.scoring.cop_loss,
                            "thief_score": config.scoring.thief_win})
    return results


def _play_series(config: Config) -> List[dict]:
    """Run a real in-process series via the orchestrator to fill the report."""
    import orchestrator
    summary = orchestrator.run(networked=False, verbose=False)
    return summary["results"]


def build_report(config: Config, play: bool = False) -> dict:
    results = _play_series(config) if play else _sample_results(config)
    report = build_internal_game_report(config, results)
    assert validate_internal(report), "generated report failed schema validation"
    return report


# ---------------------------------------------------------------------------
# Gmail delivery (optional; requires google-auth-oauthlib + credentials)
# ---------------------------------------------------------------------------
def _gmail_service(credentials_path: str, token_path: str):
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, GMAIL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"OAuth client secret not found at {credentials_path}. "
                    "Download it from Google Cloud Console (Gmail API enabled)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as fh:
            fh.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def send_email(config: Config, report: dict,
               credentials_path: str = DEFAULT_CREDENTIALS,
               token_path: str = DEFAULT_TOKEN) -> dict:
    """Send the JSON-only report via the Gmail API. Returns the API response."""
    service = _gmail_service(credentials_path, token_path)
    body = to_json(report)  # JSON only, no free text (§9)
    message = MIMEText(body, "plain", "utf-8")
    message["to"] = config.report.email_target
    message["subject"] = f"HW6 Internal Game Report - {config.report.group_name}"
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return service.users().messages().send(userId="me", body={"raw": raw}).execute()


def main():
    p = argparse.ArgumentParser(description="HW6 Gmail JSON reporter")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True,
                      help="print the JSON instead of sending (default)")
    mode.add_argument("--send", action="store_true",
                      help="actually send via the Gmail API (needs OAuth creds)")
    p.add_argument("--play", action="store_true",
                   help="play a real in-process series to populate the report")
    p.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    p.add_argument("--token", default=DEFAULT_TOKEN)
    args = p.parse_args()

    config = load_config()
    report = build_report(config, play=args.play)

    if args.send:
        print(f"Sending Internal Game JSON to {config.report.email_target} ...")
        resp = send_email(config, report, args.credentials, args.token)
        print(f"Sent. Gmail message id: {resp.get('id')}")
    else:
        print("=== Internal Game JSON (dry-run; body would be JSON only) ===")
        print(to_json(report))
        print(f"\n[dry-run] would send to: {config.report.email_target}")
        print(f"[dry-run] schema valid: {validate_internal(report)}")


if __name__ == "__main__":
    main()

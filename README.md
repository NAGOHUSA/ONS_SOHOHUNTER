# SOHO Comet Hunter (Automated)

Hourly GitHub Action that:
1) Pulls recent SOHO LASCO C2/C3 frames via Helioviewer,
2) Runs a lightweight moving-point detector to find comet-like tracks,
3) Writes PNG crops + a JSON report in `detections/`,
4) Always writes `detections/latest_status.json` + debug overlays/contacts.

![Workflow](https://github.com/NAGOHUSA/ONS_SOHOHUNTER/actions/workflows/soho-comet-hunt.yml/badge.svg)

## Quick start
- Push this repo to `main`.
- In **Actions**, run “SOHO Comet Hunt” manually once to seed status.
- Results appear under `/detections/`.  
  - Candidates: `candidates_YYYYmmdd_HHMMSS.json` + PNG crops  
  - Status (always): `latest_status.json`, `overlay_C2.png`, `overlay_C3.png`, `contact_C2.png`, `contact_C3.png`

## Optional notifications
Add `ALERT_WEBHOOK_URL` as a repo **Secret** (Slack/Discord) to get a ping when candidates are found.

## Optional AI second stage
Set **Secrets**:
- `USE_AI_CLASSIFIER` = `1`
- `OPENAI_API_KEY` = `<your key>`
Otherwise leave unset to skip paid API calls.

## Web viewer
Use the provided `soho-hunter.html`. Host it anywhere, or enable GitHub Pages and serve it from this repo.  
Edit the top CONFIG (owner/repo/branch) if needed.

## Notes
- Realtime data can have gaps; job skips gracefully.
- To increase recall temporarily, change the workflow args to `--hours 24 --step-min 15`.
- False positives happen (cosmic rays, stars/planets). Use overlays + contact sheets to verify.

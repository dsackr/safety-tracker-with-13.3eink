#!/usr/bin/env python3
# (same app as previously generated) — see prior message for full comments
import json
import os
import sqlite3
import time
from datetime import datetime, date
from pathlib import Path
from typing import List, Tuple, Dict, Any
from zoneinfo import ZoneInfo

import requests
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from PIL import Image, ImageOps, ImageDraw, ImageFont

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    APS_AVAILABLE = True
except Exception:
    APS_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
STATIC = BASE_DIR / "static"
UPLOADS = STATIC / "uploads"
THUMBS = STATIC / "thumbs"
CONVERTED = STATIC / "converted"
OSHA_DIR = STATIC / "osha"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "violations.db"
BACKGROUND_IMAGE = STATIC / "background.png"
CONFIG_PATH = DATA_DIR / "config.json"
INCIDENTS_JSON = DATA_DIR / "incidents.json"
OTHERS_JSON = DATA_DIR / "others.json"
PILLAR_KEY_PATH = DATA_DIR / "product_pillar_key.json"

LANDSCAPE_SIZE = (1600, 1200)
PORTRAIT_SIZE  = (1200, 1600)

DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DEFAULT_FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

EPD_6C_PALETTE = [(255,255,255),(0,0,0),(255,0,0),(255,255,0),(0,0,255),(0,128,0)]

DISPLAY_SCRIPT = os.environ.get("DISPLAY_SCRIPT", "")

app = Flask(__name__, static_folder=str(STATIC), template_folder=str(BASE_DIR/"templates"))
app.secret_key = os.environ.get("APP_SECRET", "change-me")

for p in [UPLOADS, THUMBS, CONVERTED, OSHA_DIR, DATA_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def db_init():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vdate TEXT NOT NULL,
                incident TEXT NOT NULL,
                vtype TEXT NOT NULL
            )
        """)
        con.commit()

def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            return {}
    return {}

def save_config(cfg: Dict[str, Any]):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

def get_api_key() -> str:
    cfg = load_config()
    return cfg.get("incident_api_key") or os.environ.get("INCIDENT_API_KEY", "")

def load_product_pillar_key() -> Dict[str, str]:
    if PILLAR_KEY_PATH.exists():
        try:
            return json.loads(PILLAR_KEY_PATH.read_text())
        except Exception:
            return {}
    return {}

def db_add_violation(vdate: str, incident: str, vtype: str):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT INTO violations (vdate, incident, vtype) VALUES (?,?,?)",
                    (vdate, incident.strip(), vtype.strip()))
        con.commit()

def db_list_violations():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT id, vdate, incident, vtype FROM violations ORDER BY vdate DESC, id DESC")
        return cur.fetchall()

def build_palette_img(colors):
    pal_img = Image.new("P", (1,1))
    pal = []
    for r,g,b in colors:
        pal.extend([r,g,b])
    pal.extend([0,0,0] * (256 - len(colors)))
    pal_img.putpalette(pal)
    return pal_img

PALETTE_IMG = build_palette_img(EPD_6C_PALETTE)

def parse_timestamp(ts: str) -> datetime:
    if not ts:
        return None
    try:
        # incident.io returns ISO strings with Z suffix
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def ny_time(dt: datetime) -> datetime:
    if dt is None:
        return None
    try:
        return dt.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        return dt

def pick_incident_time(incident: Dict[str, Any]) -> datetime:
    ts = None
    timestamps = incident.get("timestamps") or {}
    ts = timestamps.get("reported_at") or incident.get("reported_at")
    if not ts:
        ts = timestamps.get("created_at") or incident.get("created_at")
    dt = parse_timestamp(ts)
    return ny_time(dt) if dt else None

def rca_classification_from_custom_fields(incident: Dict[str, Any]) -> str:
    entries = incident.get("custom_field_entries") or incident.get("custom_fields") or []
    for entry in entries:
        name = (entry.get("name") or entry.get("custom_field", {}).get("name") or "").lower()
        cid = entry.get("custom_field_id") or entry.get("id") or ""
        if name == "rca classification" or cid == "01JZ0PNKHCB3M6NX0AHPABS59D":
            for key in ("value", "name", "value_text"):
                if entry.get(key):
                    return str(entry.get(key))
            value_option = entry.get("value_option") or {}
            for key in ("value", "name"):
                if value_option.get(key):
                    return str(value_option.get(key))
    return "Not Classified"

def extract_catalog_values(entries: list, field_name: str) -> List[str]:
    values = []
    for entry in entries:
        name = (entry.get("name") or entry.get("custom_field", {}).get("name") or "").lower()
        if name != field_name.lower():
            continue
        for key in ("values", "value_options", "value_option"):
            if entry.get(key):
                raw = entry.get(key)
                if isinstance(raw, list):
                    for item in raw:
                        val = item.get("value") or item.get("name") or item
                        if val:
                            values.append(str(val))
                elif isinstance(raw, dict):
                    val = raw.get("value") or raw.get("name")
                    if val:
                        values.append(str(val))
        for key in ("value", "name", "value_text"):
            if entry.get(key):
                values.append(str(entry.get(key)))
    return values

def extract_duration_seconds(incident: Dict[str, Any]) -> int:
    metrics = incident.get("duration_metrics") or incident.get("metrics") or []
    for metric in metrics:
        name = metric.get("name") or metric.get("label") or ""
        if name.lower() == "client impact duration":
            val = metric.get("value") or metric.get("value_seconds")
            try:
                return int(val)
            except Exception:
                continue
    custom_entries = incident.get("custom_field_entries") or []
    for entry in custom_entries:
        name = (entry.get("name") or entry.get("custom_field", {}).get("name") or "").lower()
        if name == "client impact duration":
            for key in ("value", "value_seconds", "value_int", "value_number"):
                if entry.get(key) is not None:
                    try:
                        return int(entry.get(key))
                    except Exception:
                        continue
    return None

def normalize_incident_record(incident: Dict[str, Any]) -> List[Dict[str, Any]]:
    products = extract_catalog_values(incident.get("custom_field_entries") or [], "Product")
    pillars = extract_catalog_values(incident.get("custom_field_entries") or [], "Solution Pillar")
    pillar_map = load_product_pillar_key()
    dt = pick_incident_time(incident)
    iso_time = dt.isoformat() if dt else None
    date_only = dt.date().isoformat() if dt else None
    severity_obj = incident.get("severity")
    severity = None
    if isinstance(severity_obj, dict):
        severity = severity_obj.get("name") or severity_obj.get("label")
    else:
        severity = severity_obj
    inc_type_obj = incident.get("incident_type")
    if isinstance(inc_type_obj, dict):
        event_type = inc_type_obj.get("name") or inc_type_obj.get("label") or "Operational Incident"
    else:
        event_type = inc_type_obj or "Operational Incident"
    title = incident.get("name") or incident.get("title") or incident.get("incident_title") or incident.get("reference") or incident.get("id")
    inc_number = incident.get("reference") or incident.get("id")
    rca_value = rca_classification_from_custom_fields(incident)
    duration_seconds = extract_duration_seconds(incident)
    products = products or ["Unspecified"]
    direct_pillar = pillars[0] if pillars else None
    payloads = []
    for product in products:
        pillar = pillar_map.get(product, direct_pillar)
        payloads.append({
            "inc_number": inc_number,
            "title": title,
            "event_type": event_type,
            "product": product,
            "pillar": pillar,
            "severity": severity,
            "reported_at": iso_time,
            "date": date_only,
            "rca_classification": rca_value,
            "client_impact_duration_seconds": duration_seconds,
        })
    return payloads

def upsert_payload(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = []
    idx = next((i for i, item in enumerate(data) if item.get("inc_number") == payload.get("inc_number") and item.get("product") == payload.get("product")), None)
    if idx is not None:
        data[idx] = payload
    else:
        data.append(payload)
    path.write_text(json.dumps(data, indent=2))

def sync_incident_feed(api_key: str) -> int:
    url = "https://api.incident.io/v2/incidents"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"page_size": 50}
    processed = 0
    next_cursor = None
    while True:
        if next_cursor:
            params["after"] = next_cursor
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        body = resp.json()
        inc_list = body.get("incidents") or []
        for inc in inc_list:
            payloads = normalize_incident_record(inc)
            for payload in payloads:
                target = INCIDENTS_JSON if (payload.get("event_type") or "").lower() == "operational incident" else OTHERS_JSON
                upsert_payload(target, payload)
                processed += 1
        next_cursor = body.get("pagination", {}).get("next_cursor") or body.get("pagination", {}).get("after")
        if not next_cursor:
            break
    return processed

def quantize_6c(img, dither=True):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.quantize(palette=PALETTE_IMG, dither=Image.FLOYDSTEINBERG if dither else Image.NONE).convert("RGB")

def resize_crop(img, target_size):
    tw, th = target_size
    short_target = min(tw, th)
    w, h = img.size
    if w < h:
        scale = short_target / w
    else:
        scale = short_target / h
    nw, nh = int(round(w*scale)), int(round(h*scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left = max(0, (img.width - tw)//2)
    top  = max(0, (img.height - th)//2)
    right = left + tw
    bottom = top + th
    return img.crop((left, top, right, bottom))

def letterbox(img, target_size):
    tw, th = target_size
    w, h = img.size
    if w >= h:
        scale = tw / float(w)
    else:
        scale = th / float(h)
    nw, nh = int(round(w*scale)), int(round(h*scale))
    img2 = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (tw, th), (255,255,255))
    x = (tw - nw)//2
    y = (th - nh)//2
    canvas.paste(img2, (x, y))
    return canvas

def make_thumb(src_path, size=240):
    out = THUMBS / (src_path.stem + "_thumb.jpg")
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im.thumbnail((size,size), Image.LANCZOS)
            im.save(out, "JPEG", quality=85, optimize=True)
    except Exception as e:
        print("thumb error:", e)
    return out

def process_for_display(src_path, orientation, do_crop, quantize, dither):
    target = LANDSCAPE_SIZE if orientation == "landscape" else PORTRAIT_SIZE
    out_path = CONVERTED / f"{Path(src_path).stem}_{orientation}_{'crop' if do_crop else 'nocrop'}.png"
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        out = resize_crop(img, target) if do_crop else letterbox(img, target)
        if quantize:
            out = quantize_6c(out, dither=dither)
        out.save(out_path, "PNG", optimize=True)
    return out_path

def compute_violations_state():
    def load_incident_payloads(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except Exception:
            return []

    incidents = load_incident_payloads(INCIDENTS_JSON)
    if incidents:
        def sort_key(item):
            dt = parse_timestamp(item.get("reported_at")) or parse_timestamp(item.get("date"))
            return dt or datetime.min
        incidents = sorted(incidents, key=sort_key, reverse=True)
        latest = incidents[0]
        latest_date_obj = (parse_timestamp(latest.get("reported_at")) or parse_timestamp(latest.get("date")) or datetime.now()).date()
        incident_num = latest.get("inc_number", "")
        event_type = latest.get("event_type", "")
        previous_date = None
        if len(incidents) > 1:
            prev_dt = parse_timestamp(incidents[1].get("reported_at")) or parse_timestamp(incidents[1].get("date"))
            previous_date = prev_dt.date() if prev_dt else None
        today = date.today()
        days_since = (today - latest_date_obj).days
        prior_count = (latest_date_obj - previous_date).days if previous_date else None
        return {
            "days_since": days_since,
            "prior_count": prior_count,
            "incident": incident_num,
            "vtype": event_type,
            "latest_date": latest_date_obj,
            "previous_date": previous_date,
            "latest_incident": latest,
        }
    rows = db_list_violations()
    if not rows:
        return {"days_since": None, "prior_count": None, "incident": "", "vtype": "", "latest_date": None, "previous_date": None, "latest_incident": None}
    latest = rows[0]
    latest_date = datetime.strptime(latest[1], "%Y-%m-%d").date()
    incident = latest[2]
    vtype = latest[3]
    previous_date = None
    if len(rows) > 1:
        previous_date = datetime.strptime(rows[1][1], "%Y-%m-%d").date()
    today = date.today()
    days_since = (today - latest_date).days
    prior_count = (latest_date - previous_date).days if previous_date else None
    return {"days_since": days_since, "prior_count": prior_count, "incident": incident, "vtype": vtype,
            "latest_date": latest_date, "previous_date": previous_date, "latest_incident": None}

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        return ImageFont.load_default()

def render_osha_sign(state, out_path):
    W, H = 1200, 1600
    if BACKGROUND_IMAGE.exists():
        base = Image.open(BACKGROUND_IMAGE).convert("RGB").resize((W,H), Image.LANCZOS)
    else:
        base = Image.new("RGB", (W,H), (255,223,102))
    draw = ImageDraw.Draw(base)
    f_days   = _load_font(DEFAULT_FONT, 460)
    f_small  = _load_font(DEFAULT_FONT, 120)
    f_medium = _load_font(DEFAULT_FONT, 90)
    if state["days_since"] is not None:
        txt = str(state["days_since"])
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), txt, font=f_days)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            w, h = draw.textsize(txt, font=f_days)
        draw.text(((W-w)//2, 460), txt, font=f_days, fill=(0,0,0))
    prior = "—" if state["prior_count"] is None else str(state["prior_count"])
    draw.text((70, 1220), "PRIOR COUNT:", font=f_medium, fill=(255,255,255))
    draw.text((100, 1310), prior, font=f_small, fill=(255,255,255))
    draw.text((430, 1220), "OFFENDING INCIDENT:", font=f_medium, fill=(255,255,255))
    inc_txt = f"INC-{state['incident']}" if state['incident'] else "INC-"
    draw.text((500, 1310), inc_txt, font=f_small, fill=(255,255,255))
    check_x = 1000
    change_y = 1220
    deploy_y = 1320
    missed_y = 1420
    box_size = 40
    def box(x,y,label,checked):
        draw.rectangle([x, y, x+box_size, y+box_size], outline=(255,255,255), width=4)
        if checked:
            draw.line([x+8,y+22, x+18,y+35, x+35,y+8], fill=(0,0,255), width=6)
        draw.text((x+box_size+10, y-10), label, font=_load_font(DEFAULT_FONT, 60), fill=(255,255,255))
    vt = (state["vtype"] or "").lower()
    box(check_x, change_y, "Change", vt=="change")
    box(check_x, deploy_y, "Deploy", vt=="deploy")
    box(check_x, missed_y, "Missed", vt=="missed")
    base.save(out_path, "PNG", optimize=True)
    return out_path

def build_and_get_osha_preview():
    state = compute_violations_state()
    out_path = OSHA_DIR / "current.png"
    render_osha_sign(state, out_path)
    return out_path

def send_to_display(image_path, orientation="portrait"):
    try:
        from PIL import Image
        # import your driver
        from epd13in3E import EPD  # adjust name if file is different

        img = Image.open(image_path).convert("RGB")

        epd = EPD()
        epd.Init()
        try:
            buf = epd.getbuffer(img)   # your driver builds the 4-bit packed buffer
            epd.display(buf)
        finally:
            epd.sleep()

        return True, "Panel updated."
    except Exception as e:
        return False, f"EPD error: {e}"

sched = None
def _schedule_daily_push():
    if not APS_AVAILABLE:
        return
    global sched
    if sched is None:
        sched = BackgroundScheduler(daemon=True)
        def job():
            try:
                path = build_and_get_osha_preview()
                ok, msg = send_to_display(path, "portrait")
                print(f"[daily 06:00] Pushed OSHA image: ok={ok}, msg={msg}")
            except Exception as e:
                print(f"[daily 06:00] Job error: {e}")
        sched.add_job(job, "cron", hour=6, minute=0)
        sched.start()

@app.route("/config", methods=["GET", "POST"])
def config_page():
    cfg = load_config()
    current_key = cfg.get("incident_api_key", "")
    if request.method == "POST":
        api_key = request.form.get("incident_api_key", "").strip()
        if api_key:
            cfg["incident_api_key"] = api_key
            save_config(cfg)
            flash("Incident.io API key saved.", "ok")
        else:
            cfg.pop("incident_api_key", None)
            save_config(cfg)
            flash("API key cleared.", "ok")
        return redirect(url_for("config_page"))
    if current_key:
        masked = ("•" * (len(current_key) - 4)) + current_key[-4:] if len(current_key) > 4 else "•" * len(current_key)
    else:
        masked = ""
    return render_template("config.html", api_key_masked=masked, has_key=bool(current_key))

@app.route("/")
def index():
    db_init()
    build_and_get_osha_preview()
    images = []
    for p in sorted(UPLOADS.glob("*")):
        if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".gif",".webp"):
            thumb = make_thumb(p)
            images.append({
                "name": p.name,
                "thumb_url": url_for("static", filename=f"thumbs/{thumb.name}"),
                "src_url": url_for("static", filename=f"uploads/{p.name}"),
            })
    state = compute_violations_state()
    incidents_synced = INCIDENTS_JSON.exists()
    return render_template("index.html",
                           images=images,
                           osha_url=url_for("static", filename="osha/current.png") + f"?cb={{int(time.time())}}",
                            has_bg=BACKGROUND_IMAGE.exists(),
                            latest_incident=state.get("latest_incident"),
                            api_configured=bool(get_api_key()),
                            incidents_synced=incidents_synced)

@app.post("/incidents/sync")
def sync_incidents():
    api_key = get_api_key()
    if not api_key:
        flash("Configure an incident.io API key first.", "error")
        return redirect(url_for("config_page"))
    try:
        count = sync_incident_feed(api_key)
        build_and_get_osha_preview()
        flash(f"Synced {count} incident entries from incident.io.", "ok")
    except Exception as e:
        flash(f"Sync failed: {e}", "error")
    return redirect(url_for("index")+"#violations")

@app.post("/upload")
def upload():
    f = request.files.get("image")
    orientation = request.form.get("orientation", "portrait")
    do_crop = request.form.get("method","crop") == "crop"
    quantize = request.form.get("quantize") == "on"
    dither = request.form.get("dither") == "on"
    if not f or f.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index")+"#images")
    dst = UPLOADS / Path(f.filename).name
    f.save(dst)
    try:
        out = process_for_display(dst, orientation, do_crop, quantize, dither)
    except Exception as e:
        flash(f"Process error: {e}", "error")
        return redirect(url_for("index")+"#images")
    flash("Image uploaded and processed.", "ok")
    return redirect(url_for("preview", filename=out.name))

@app.get("/preview/<filename>")
def preview(filename):
    return render_template("preview.html",
                           img_url=url_for("static", filename=f"converted/{filename}") + f"?cb={{int(time.time())}}",
                           filename=filename)

@app.post("/send/<filename>")
def send(filename):
    orientation = request.form.get("orientation", "portrait")
    path = CONVERTED / filename
    ok, msg = send_to_display(path, orientation)
    flash(("Sent to display." if ok else "Failed to send.") + f" {msg}", "ok" if ok else "error")
    return redirect(url_for("preview", filename=filename))

@app.post("/violations/report")
def violations_report():
    vdate = request.form.get("vdate", "")
    incident = request.form.get("incident", "").strip()
    vtype = request.form.get("vtype", "Change")
    try:
        datetime.strptime(vdate, "%Y-%m-%d")
    except Exception:
        flash("Invalid date.", "error")
        return redirect(url_for("index")+"#violations")
    if not incident:
        flash("Incident number required.", "error")
        return redirect(url_for("index")+"#violations")
    db_add_violation(vdate, incident, vtype)
    out = build_and_get_osha_preview()
    push = request.form.get("push_display") == "on"
    if push:
        ok, msg = send_to_display(out, "portrait")
        flash(("OSHA image sent to display." if ok else "Failed to send OSHA image.") + f" {msg}", "ok" if ok else "error")
    else:
        flash("OSHA image updated on the web preview. Use the button to push to the display.", "ok")
    return redirect(url_for("index")+"#violations")

@app.post("/violations/push")
def violations_push():
    out = build_and_get_osha_preview()
    ok, msg = send_to_display(out, "portrait")
    flash(("OSHA image sent to display." if ok else "Failed to send OSHA image.") + f" {msg}", "ok" if ok else "error")
    return redirect(url_for("index")+"#violations")

@app.get("/converted/<path:filename>")
def converted(filename):
    return send_from_directory(CONVERTED, filename, as_attachment=False)

if __name__ == "__main__":
    db_init()
    try:
        build_and_get_osha_preview()
    except Exception as e:
        print("Initial OSHA render error:", e)
    _schedule_daily_push()
    app.run(host="0.0.0.0", port=5001, debug=False)

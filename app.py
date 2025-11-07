#!/usr/bin/env python3
# (same app as previously generated) — see prior message for full comments
import os
import io
import sqlite3
import time
from datetime import datetime, date
from pathlib import Path
from typing import List, Tuple

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
    rows = db_list_violations()
    if not rows:
        return {"days_since": None, "prior_count": None, "incident": "", "vtype": "", "latest_date": None, "previous_date": None}
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
            "latest_date": latest_date, "previous_date": previous_date}

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
        w,h = draw.textsize(txt, font=f_days)
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
    if DISPLAY_SCRIPT and Path(DISPLAY_SCRIPT).exists():
        try:
            import subprocess
            cmd = ["python3", DISPLAY_SCRIPT, str(image_path), orientation]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            ok = proc.returncode == 0
            msg = proc.stdout if ok else (proc.stderr or "Display script failed")
            return ok, msg
        except Exception as e:
            return False, f"Error calling display script: {e}"
    else:
        return True, "Simulated push to display."

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

@app.route("/")
def index():
    db_init()
    osha_path = build_and_get_osha_preview()
    images = []
    for p in sorted(UPLOADS.glob("*")):
        if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".gif",".webp"):
            thumb = make_thumb(p)
            images.append({
                "name": p.name,
                "thumb_url": url_for("static", filename=f"thumbs/{thumb.name}"),
                "src_url": url_for("static", filename=f"uploads/{p.name}"),
            })
    return render_template("index.html",
                           images=images,
                           osha_url=url_for("static", filename="osha/current.png") + f"?cb={{int(time.time())}}",
                           has_bg=BACKGROUND_IMAGE.exists())

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

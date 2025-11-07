# Safety Tracker with 13.3" E-Ink Display

A Flask web application for managing safety incident reports and preparing imagery tailored for a 13.3" E-Ink display. The app keeps track of OSHA-style "Days Since Last Incident" signage, converts uploaded images to match the display's palette, and can optionally push generated assets to external hardware via a custom script.

## Features
- Upload images and generate resized/quantized versions optimized for landscape or portrait orientations.
- Maintain a history of safety violations and automatically render an OSHA poster showing days since the last incident.
- Preview generated images from the browser and optionally send them to an external display controller.
- Optional daily scheduled push to hardware when `apscheduler` is available.

## Requirements
- Python 3.9+
- [Flask](https://flask.palletsprojects.com/)
- [Pillow](https://python-pillow.org/)
- Optional: [APScheduler](https://apscheduler.readthedocs.io/) (for scheduled pushes)

Install dependencies with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install Flask Pillow APScheduler
```

## Configuration
- **`DISPLAY_SCRIPT`** (optional): path to a Python script used to push generated images to your display hardware. When unset, the app simulates the push.
- **`APP_SECRET`** (optional): secret key for Flask session management. Defaults to `change-me`.

## Running the App
1. Ensure the virtual environment is activated and dependencies are installed.
2. Initialize the SQLite database and start the development server:

   ```bash
   python app.py
   ```

3. Open `http://localhost:5001` in your browser.

The application creates these directories automatically if they do not exist:

- `static/uploads` — original uploaded images.
- `static/thumbs` — thumbnails used in the gallery.
- `static/converted` — processed images sized for the display.
- `static/osha` — the current OSHA poster render.
- `data/` — SQLite database files (e.g., `violations.db`).

## Usage Tips
- Upload high-resolution images for best results. Choose between "Crop" (fills the frame) and "Letterbox" (adds padding) processing methods.
- Toggle quantization and dithering to preview the final appearance on the limited E-Ink palette.
- Record safety violations with dates, incident numbers, and type (Change/Deploy/Missed). The OSHA poster updates immediately, and you can optionally push it to the hardware.
- If APScheduler is installed, the app schedules a daily 06:00 push of the OSHA image when running as the main module.

## Development
- Templates live in `templates/` and static assets (CSS, images) in `static/`.
- Background image for the OSHA sign can be customized by replacing `static/background.png`.
- Run the Flask app with `FLASK_ENV=development` if you prefer automatic reloads:

  ```bash
  export FLASK_ENV=development
  flask --app app run --port 5001
  ```

## License
This project does not currently include a license. Add one if you plan to distribute or open-source your modifications.

import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Crop Doctor", page_icon="🌱", layout="centered")

st.markdown("""
<style>
  .block-container {padding-top: 1.5rem; max-width: 720px;}
  div.stButton > button, div.stDownloadButton > button {font-size: 1.15rem; padding: 0.7rem 1rem; border-radius: 12px;}
  .step {font-size: 1.15rem; font-weight: 700; margin: 1.1rem 0 0.3rem;}
  .verdict {border-radius: 14px; padding: 1rem 1.2rem; margin: 0.5rem 0 1rem; font-size: 1.1rem;}
  .verdict h3 {margin: 0 0 0.3rem; font-size: 1.4rem;}
  .v-red {background: rgba(211,47,47,.12); border-left: 6px solid #d32f2f;}
  .v-green {background: rgba(46,125,50,.12); border-left: 6px solid #2e7d32;}
  .v-amber {background: rgba(237,108,2,.12); border-left: 6px solid #ed6c02;}
  .spray {border-radius: 14px; padding: 0.9rem 1.1rem; margin: 0.6rem 0; font-size: 1.05rem;}
</style>
""", unsafe_allow_html=True)

PLACES = {
    "Chennai": (13.08, 80.27), "Coimbatore": (11.02, 76.96), "Madurai": (9.93, 78.12),
    "Tiruchirappalli": (10.79, 78.70), "Salem": (11.66, 78.15), "Tirunelveli": (8.71, 77.76),
    "Erode": (11.34, 77.72), "Vellore": (12.92, 79.13), "Thanjavur": (10.79, 79.14),
    "Bengaluru": (12.97, 77.59), "Hyderabad": (17.39, 78.49), "Pune": (18.52, 73.86),
    "New Delhi": (28.61, 77.21),
}
OTHER_PLACE, NO_PLACE = "Other place (enter coordinates)", "Skip the weather check"

CROP_ICONS = {"Apple": "🍎", "Blueberry": "🫐", "Cherry (including sour)": "🍒", "Corn (maize)": "🌽",
              "Grape": "🍇", "Orange": "🍊", "Peach": "🍑", "Pepper, bell": "🫑", "Potato": "🥔",
              "Raspberry": "🍓", "Soybean": "🫘", "Squash": "🎃", "Strawberry": "🍓", "Tomato": "🍅"}


@st.cache_data(ttl=3600, show_spinner=False)
def get_crops():
    return requests.get(f"{API_URL}/crops", timeout=10).json()


def wait_for_api(seconds: int = 30) -> bool:
    for _ in range(seconds):
        try:
            if requests.get(f"{API_URL}/health", timeout=2).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def disease_name(label: str) -> str:
    crop, disease = label.split("___", 1)
    disease = disease.replace("_", " ").strip()
    return "Healthy" if disease.lower() == "healthy" else disease


def crop_name(label: str) -> str:
    return label.split("___", 1)[0].replace("_", " ").strip()


def sureness(p: float) -> str:
    return "very sure" if p >= 0.9 else "fairly sure" if p >= 0.7 else "not sure"


def friendly_error(r: requests.Response) -> str:
    detail = ""
    try:
        detail = r.json().get("detail", "")
    except ValueError:
        pass
    if r.status_code == 413:
        return "That photo is too large. Please take a new photo or choose a smaller one (under 10 MB)."
    if r.status_code in (400, 415):
        return f"I couldn't read that photo. Please use a normal JPG or PNG picture of a leaf. ({detail})"
    return f"Something went wrong ({r.status_code}). Please try again. {detail}"


def show_result(res: dict):
    top = res["predictions"][0]
    d = res["decision"]

    if d == "advise":
        cls, icon = "v-red", "🦠"
        title = f"{disease_name(top['label'])} on {crop_name(top['label'])}"
        body = f"I'm {sureness(top['prob'])} ({top['prob']:.0%}). See what to do below."
    elif d == "healthy":
        cls, icon, title = "v-green", "✅", "Your leaf looks healthy"
        body = f"I'm {sureness(top['prob'])} ({top['prob']:.0%}). No treatment needed. Check again if spots appear."
    elif d == "ambiguous_pair":
        a, b = res["predictions"][0], res["predictions"][1]
        cls, icon, title = "v-amber", "🤔", "Two diseases look alike here"
        body = (f"It could be <b>{disease_name(a['label'])}</b> or <b>{disease_name(b['label'])}</b>. "
                "They need different treatment, so please don't spray yet. Show the plant to your "
                "agriculture officer, or take a closer photo of a spot.")
    else:
        cls, icon, title = "v-amber", "📷", "Please take another photo"
        if res.get("reason") == "crop_mismatch":
            body = ("This doesn't look like the crop you picked. Check the crop in Step 1, "
                    "or take a clearer photo of a single leaf.")
        else:
            body = "I'm not sure enough to suggest a treatment. A clearer photo will help."

    st.markdown(f'<div class="verdict {cls}"><h3>{icon} {title}</h3>{body}</div>', unsafe_allow_html=True)

    if d == "uncertain_retake_photo":
        st.markdown("**Tips for a good photo:** one leaf fills the picture · daylight, no shadow · "
                    "hold steady so it's sharp · show the spots clearly")

    w = res.get("weather")
    if w:
        rain = w.get("rain_prob_next_6h")
        if w.get("modified_advice"):
            st.markdown(f'<div class="spray v-amber">🌧️ <b>Don\'t spray right now.</b> Rain is likely in the '
                        f'next 6 hours ({rain:.0%} chance) and would wash the spray off. Spray after the rain.</div>',
                        unsafe_allow_html=True)
        elif rain is None:
            st.markdown('<div class="spray v-amber">❔ <b>Couldn\'t check the weather.</b> Don\'t spray if rain '
                        'is expected in the next 6 hours.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="spray v-green">☀️ <b>Good time to spray.</b> Low chance of rain in the '
                        f'next 6 hours ({rain:.0%}).</div>', unsafe_allow_html=True)

    if res.get("advice"):
        st.markdown("### What to do")
        st.markdown(res["advice"])

    if res.get("sources"):
        with st.expander("📚 Where this advice comes from"):
            for s in res["sources"]:
                st.markdown(f"- [{s['title']}]({s['url']}): {s['publisher']} ({s['region']})")

    alert = res.get("alert", {}).get("status", {})
    if alert.get("telegram") == "sent":
        st.success("📨 Sent to your Telegram.")
    elif "telegram" in alert:
        st.warning("Couldn't send to Telegram this time. The result is saved here.")

    with st.expander("🔎 Other possibilities"):
        for p in res["predictions"]:
            st.progress(min(max(p["prob"], 0.0), 1.0), text=f"{crop_name(p['label'])}: {disease_name(p['label'])} ({p['prob']:.1%})")
        if res.get("crop_selected") and res["predictions_raw"][0]["label"] != top["label"]:
            st.caption(f"Without your crop choice, the model's first guess was {crop_name(res['predictions_raw'][0]['label'])}: "
                       f"{disease_name(res['predictions_raw'][0]['label'])}.")

    if st.button("🔄 Check another leaf", use_container_width=True):
        for k in ("result", "photo_bytes"):
            st.session_state.pop(k, None)
        st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1
        st.rerun()


def render_history():
    st.header("📒 History")
    try:
        rows = requests.get(f"{API_URL}/detections", params={"limit": 200}, timeout=10).json()
    except requests.RequestException:
        rows = []
    if not rows:
        st.info("No plants checked yet. Your checks will appear here.")
    else:
        df = pd.DataFrame(rows)
        df["When"] = pd.to_datetime(df["ts"]).dt.tz_convert("Asia/Kolkata").dt.strftime("%d %b, %I:%M %p")
        df["Result"] = df["top1"].fillna("").map(lambda l: f"{crop_name(l)}: {disease_name(l)}" if l else "")
        df["Sure"] = df["p1"].map(lambda p: f"{p:.0%}" if pd.notna(p) else "")
        df["Outcome"] = df["decision"].map({"advise": "Treatment advised", "healthy": "Healthy",
                                            "uncertain_retake_photo": "Asked for new photo",
                                            "ambiguous_pair": "Two look-alikes"}).fillna(df["decision"])
        df["Spray"] = df["delay_spraying"].map({1: "Wait for rain to pass", 0: ""}).fillna("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Plants checked", len(df))
        c2.metric("Diseases found", int((df["decision"] == "advise").sum()))
        c3.metric("Told to wait for rain", int(df["delay_spraying"].fillna(0).sum()))
        st.dataframe(df[["When", "Result", "Sure", "Outcome", "Spray"]], use_container_width=True, hide_index=True)
        with st.expander("Chart: outcomes"):
            st.bar_chart(df["Outcome"].value_counts())


tab_check, tab_history = st.tabs(["🌿 Check my plant", "📒 History"])

with tab_history:
    render_history()

with tab_check:
    st.title("🌱 Crop Doctor")
    st.write("Find out what's wrong with your plant and what to do, in four quick steps.")

    try:
        crops = get_crops()
    except requests.RequestException:
        with st.spinner("Starting up the diagnosis service..."):
            ok = wait_for_api()
        if not ok:
            st.error("The diagnosis service isn't running. Start the app with **run.ps1** "
                     "(it starts everything), then refresh this page.")
            st.stop()
        crops = get_crops()

    nonce = st.session_state.get("uploader_nonce", 0)

    if "result" in st.session_state:
        st.image(st.session_state["photo_bytes"], width=180)
        show_result(st.session_state["result"])
        st.caption("⚠️ This is a student prototype for 14 crops. It is not a certified agricultural tool. "
                   "Always follow the pesticide label and ask your local agriculture officer before spraying.")
        st.stop()

    st.markdown('<div class="step">1️⃣ Which crop is it?</div>', unsafe_allow_html=True)
    options = [c["label"] for c in crops] + ["I'm not sure"]
    choice = st.selectbox("Crop", options, index=options.index("Tomato") if "Tomato" in options else 0,
                          format_func=lambda o: f"{CROP_ICONS.get(o, '❓')}  {o}", label_visibility="collapsed")
    crop_value = next((c["value"] for c in crops if c["label"] == choice), None)

    st.markdown('<div class="step">2️⃣ Take or upload a photo of one leaf</div>', unsafe_allow_html=True)
    up_tab, cam_tab = st.tabs(["🖼️ Photo from phone / gallery", "📷 Use webcam"])
    with up_tab:
        up = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"], key=f"up{nonce}",
                              label_visibility="collapsed")
    with cam_tab:
        cam = st.toggle("Turn on the webcam", key=f"camon{nonce}") and \
              st.camera_input("Take a photo", key=f"cam{nonce}", label_visibility="collapsed")
    photo = up or cam or None
    if photo:
        try:
            st.image(photo, width=240)
        except Exception:
            st.error("I couldn't open that photo. Please choose a normal JPG or PNG picture of a leaf.")
            photo = None
    st.caption("Tip: one leaf filling the picture, in daylight, works best.")

    st.markdown('<div class="step">3️⃣ Where is your farm? (to check rain before spraying)</div>', unsafe_allow_html=True)
    place = st.selectbox("Place", list(PLACES) + [OTHER_PLACE, NO_PLACE], label_visibility="collapsed")
    lat = lon = None
    if place in PLACES:
        lat, lon = PLACES[place]
    elif place == OTHER_PLACE:
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=13.08, min_value=-90.0, max_value=90.0, format="%.4f")
        lon = c2.number_input("Longitude", value=80.27, min_value=-180.0, max_value=180.0, format="%.4f")
        st.caption("Find these on Google Maps: long-press your field and copy the two numbers.")

    send_tg = st.toggle("📨 Also send the result to my Telegram", value=False)

    st.markdown('<div class="step">4️⃣ Check my plant</div>', unsafe_allow_html=True)
    go = st.button("🔍 Check my plant", type="primary", use_container_width=True, disabled=photo is None)
    if photo is None:
        st.caption("Add a photo in Step 2 to continue.")

    if go and photo:
        data = {"crop": crop_value or "", "channels": "telegram" if send_tg else ""}
        if lat is not None:
            data.update(lat=str(lat), lon=str(lon))
        with st.status("Checking your plant...", expanded=True) as status:
            st.write("🔬 Looking at the leaf...")
            try:
                r = requests.post(f"{API_URL}/predict", timeout=120, data=data,
                                  files={"image": (getattr(photo, "name", None) or "photo.jpg",
                                                   photo.getvalue(), photo.type or "image/jpeg")})
            except requests.RequestException:
                status.update(label="Couldn't reach the diagnosis service", state="error")
                st.error("The diagnosis service didn't answer. Check that run.ps1 is still running, then try again.")
                st.stop()
            if r.status_code != 200:
                status.update(label="Problem with the photo", state="error")
                st.error(friendly_error(r))
                st.stop()
            st.write("📚 Finding advice from agricultural guides...")
            st.write("🌦️ Checking the weather...")
            status.update(label="Done!", state="complete", expanded=False)
        st.session_state["result"] = r.json()
        st.session_state["photo_bytes"] = photo.getvalue()
        st.rerun()

    st.caption("⚠️ This is a student prototype for 14 crops. It is not a certified agricultural tool. "
               "Always follow the pesticide label and ask your local agriculture officer before spraying.")


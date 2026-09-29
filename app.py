import io
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("best.pt")  # YOLOv8n-cls: classes = healthy / unhealthy

HTML = """
<!doctype html>
<html lang="hi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Onion Health Checker</title>
<style>
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,sans-serif;background:#f6f1ea;color:#222;display:flex;justify-content:center;padding:20px}
  .card{background:#fff;max-width:460px;width:100%;border-radius:18px;padding:24px;box-shadow:0 6px 24px rgba(0,0,0,.08)}
  h1{font-size:1.4rem;margin:0 0 4px}
  p.sub{margin:0 0 18px;color:#666;font-size:.9rem}
  .drop{border:2px dashed #c9a27a;border-radius:14px;padding:26px;text-align:center;cursor:pointer;background:#fffaf4}
  .drop img{max-width:100%;max-height:260px;border-radius:10px}
  button{width:100%;margin-top:14px;padding:14px;border:0;border-radius:12px;background:#8e3b8f;color:#fff;font-size:1rem;cursor:pointer}
  button:disabled{opacity:.5}
  .result{margin-top:18px;padding:16px;border-radius:12px;text-align:center;display:none}
  .healthy{background:#e3f6e5;color:#1b6b2a}
  .unhealthy{background:#fde6e4;color:#a12622}
  .result h2{margin:0;font-size:1.5rem}
  .bar{height:8px;background:#0002;border-radius:4px;margin-top:10px;overflow:hidden}
  .bar div{height:100%;background:currentColor}
</style>
</head>
<body>
<div class="card">
  <h1>🧅 Onion Health Checker</h1>
  <p class="sub">Pyaz ki photo daaliye, model batayega healthy hai ya unhealthy.</p>
  <label class="drop" id="drop">
    <input type="file" id="file" accept="image/*" hidden>
    <div id="preview">📷 Photo chuniye ya kheenchiye</div>
  </label>
  <button id="btn" disabled>Check karo</button>
  <div class="result" id="result">
    <h2 id="label"></h2>
    <div id="conf"></div>
    <div class="bar"><div id="barfill"></div></div>
  </div>
</div>
<script>
const file=document.getElementById('file'),btn=document.getElementById('btn'),
      prev=document.getElementById('preview'),res=document.getElementById('result');
file.onchange=()=>{
  const f=file.files[0]; if(!f) return;
  prev.innerHTML='<img src="'+URL.createObjectURL(f)+'">';
  btn.disabled=false; res.style.display='none';
};
btn.onclick=async()=>{
  const fd=new FormData(); fd.append('image',file.files[0]);
  btn.disabled=true; btn.textContent='Check ho raha hai...';
  try{
    const r=await fetch('/predict',{method:'POST',body:fd});
    const d=await r.json();
    if(d.error) throw new Error(d.error);
    res.className='result '+d.label; res.style.display='block';
    document.getElementById('label').textContent=d.label==='healthy'?'✅ Healthy':'⚠️ Unhealthy';
    document.getElementById('conf').textContent='Confidence: '+d.confidence+'%';
    document.getElementById('barfill').style.width=d.confidence+'%';
  }catch(e){alert('Error: '+e.message)}
  btn.disabled=false; btn.textContent='Check karo';
};
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify(error="Koi image nahi mili"), 400
    try:
        img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    except Exception:
        return jsonify(error="Image padh nahi paye"), 400
    r = model.predict(img, verbose=False)[0]
    idx = int(r.probs.top1)
    return jsonify(
        label=r.names[idx],
        confidence=round(float(r.probs.top1conf) * 100, 2),
        probabilities={r.names[i]: round(float(p) * 100, 2) for i, p in enumerate(r.probs.data.tolist())},
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

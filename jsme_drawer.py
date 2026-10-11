# jsme_drawer.py
import streamlit.components.v1 as components

def show_jsme_drawer(height: int = 660) -> None:
    """
    Render a standalone JSME drawing widget inside Streamlit.
    - The editor provides 'Copy SMILES' (copies SMILES to clipboard or shows a prompt).
    - The parent Streamlit app is NOT automatically modified by this component.
    - Users should click 'Copy SMILES' then use the parent 'Paste into search' button
      to move SMILES into the app's main search box.
    """
    js = r"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Sketch → SMILES (JSME)</title>
      <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/jsme-editor/jsme.nocache.js"></script>
      <style>
        body { font-family: Arial, sans-serif; margin: 12px; }
        #appletContainer { border: 1px solid #ccc; width: 640px; height: 360px; }
        #result { margin-top: 12px; word-break: break-all; }
        #controls { margin-top: 8px; }
        button { padding: 8px 12px; margin-right: 8px; }
        .note { color: #666; font-size: 13px; margin-top:6px; }
      </style>
    </head>
    <body>
      <h3>Draw a molecule</h3>
      <div id="appletContainer"></div>

      <div id="controls">
        <button id="btnCopy">Copy SMILES</button>
        <button id="btnPrompt">Show SMILES (prompt)</button>
        <button id="btnClear">Clear</button>
        <span id="status" style="margin-left:10px;color:#444;"></span>
      </div>

      <div id="result" class="note">Use "Copy SMILES" then click "Paste SMILES into search" in the main app.</div>

    <script>
      function jsmeOnLoad() {
          try {
            window.jsmeApplet = new JSApplet.JSME("appletContainer", "640px", "360px", {options: "saveButton:false"});
            document.getElementById('status').textContent = 'Editor ready';
          } catch (e) {
            document.getElementById('status').textContent = 'Error initializing JSME: ' + e;
          }
      }

      function getSmilesFromApplet() {
        try {
          if (window.jsmeApplet && typeof window.jsmeApplet.smiles === 'function') {
            return window.jsmeApplet.smiles() || "";
          }
        } catch (e) {
          console.log("JSME get error", e);
        }
        return "";
      }

      function copyToClipboard(text) {
        if (!text) return;
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(function(){
            // confirmation visible to user
            document.getElementById('result').textContent = "SMILES copied to clipboard: " + text;
          }, function(err){
            prompt("Copy SMILES (use Ctrl+C):", text);
          });
        } else {
          prompt("Copy SMILES (use Ctrl+C):", text);
        }
      }

      document.addEventListener('DOMContentLoaded', function() {
        document.getElementById('btnCopy').onclick = function() {
          var s = getSmilesFromApplet();
          if (!s) {
            alert("No structure present (draw something first).");
            return;
          }
          copyToClipboard(s);
        };

        document.getElementById('btnPrompt').onclick = function() {
          var s = getSmilesFromApplet() || "";
          prompt("SMILES (copy with Ctrl+C):", s);
        };

        document.getElementById('btnClear').onclick = function() {
          try {
            if (window.jsmeApplet && typeof window.jsmeApplet.readMolecule === 'function') {
              window.jsmeApplet.readMolecule('');
            }
            document.getElementById('result').textContent = "Cleared.";
          } catch (e) { console.log('clear error', e); }
        };
      });

      if (typeof JSApplet !== 'undefined') {
        try { jsmeOnLoad(); } catch (e) { console.log(e); }
      }
    </script>
    </body>
    </html>
    """
    components.html(js, height=height, scrolling=True)

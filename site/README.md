# TUD-Station Showcase

Interactive documentation website for the MSc thesis:  
**LLM Navigation Agent for Urban Rail Transit Stations**  
TU Delft · MSc Architecture · 2025

🔗 **Live site**: [https://YOUR_USERNAME.github.io/tud-station-showcase](https://YOUR_USERNAME.github.io/tud-station-showcase)

---

## Overview

This repository hosts a static GitHub Pages website that documents the complete TUD-Station research project — an end-to-end system from IFC BIM parsing to a conversational LLM navigation agent.

**Key metrics:**

- 18,648 graph nodes · 68,148 edges
- 98.5% agent arrival rate
- 160.2 s average travel time
- 200 Mesa ABM agents across 3 scenarios
- DeepSeek function calling + ChromaDB RAG + FAISS node index


---

## Site Structure

```
site/
├── index.html              ← Landing page (hero + metrics + navigation)
├── pipeline.html           ← Six-step pipeline walkthrough
├── navgraph.html           ← 2.5D navigation graph construction
├── simulation.html         ← ABM simulation (3 scenarios + interactive)
├── agent.html              ← LLM Agent (tools, RAG, FAISS, examples)
├── results.html            ← Evaluation metrics + image gallery
├── assets/
│   ├── css/style.css       ← Shared design system
│   └── js/nav.js           ← Navigation, lightbox, counters, timeline
├── interactive/            ← 9 interactive Plotly HTML visualisations
│   ├── interactive_graph.html
│   ├── interactive_cross_section.html
│   ├── interactive_station_3d.html
│   ├── interactive_3d_sim_static.html
│   ├── interactive_3d_sim_dynamic.html
│   ├── interactive_sim_static.html
│   ├── interactive_sim_dynamic.html
│   ├── interactive_entrance_routes.html
│   └── interactive_route_diff.html
└── img/                    ← 40+ output figures (PNG)
```

---

## Deploying to GitHub Pages

### Step 1 — Create a new GitHub repository

```bash
# On GitHub.com: New repository → name it "tud-station-showcase"
# Visibility: Public (required for free GitHub Pages)
```

### Step 2 — Copy site assets into this repository

From your thesis workspace, copy:

```bash
# From the thesis repo:
cp -r station/site/* .
cp -r station/experiment/docs/methodology/interactive/* interactive/
cp -r station/experiment/docs/methodology/img/* img/
```

### Step 3 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial showcase site"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/tud-station-showcase.git
git push -u origin main
```

### Step 4 — Enable GitHub Pages

1. Go to repository **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: **main** · Folder: **/ (root)**
4. Click **Save**

The site will be live at `https://YOUR_USERNAME.github.io/tud-station-showcase` within ~60 seconds.

---

## Local Preview

No build step required. Open any HTML file directly in a browser, or use a local server:

```bash
# Python (recommended — needed for iframe src resolution)
python -m http.server 8080
# → open http://localhost:8080
```

---

## Adding / Updating Content

| What to update | Where |
| --- | --- |
| Text content | Edit the relevant `.html` file |
| Styles / colours | `assets/css/style.css` |
| Navigation behaviour | `assets/js/nav.js` |
| Interactive visualisations | Replace files in `interactive/` |
| Output images | Replace files in `img/` |

---

## Tech Stack

- Pure HTML + CSS + JavaScript (no build tools, no dependencies)
- [Plotly](https://plotly.com) — interactive visualisation (bundled in `interactive/`)
- GitHub Pages — static hosting

---

## License

This website and its content are part of an academic thesis. Please contact the author before reuse.

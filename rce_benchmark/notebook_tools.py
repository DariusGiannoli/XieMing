"""Notebook helpers for visual-first RCE evaluation."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np

from rce_benchmark.annotations import (
    annotation_to_serializable,
    bbox_from_mask,
    mask_from_polygon,
    save_mask,
)
from rce_benchmark.config import load_config
from rce_benchmark.datasets import load_episode_assets, load_episode_manifest
from rce_benchmark.tasks import run_stage1_speed, run_stage2_middlebury
from rce_benchmark.types import Annotation, Episode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_POLYGONS: dict[str, list[list[int]]] = {}
NOTEBOOK_BOXES: dict[str, list[int]] = {}


def _display_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _polygon_from_bbox(bbox: list[int] | None) -> list[list[int]]:
    if bbox is None:
        return []
    x0, y0, x1, y1 = map(int, bbox)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _annotation_polygon(annotation: Annotation) -> list[list[int]]:
    if annotation.polygon:
        return [[int(x), int(y)] for x, y in annotation.polygon]
    return _polygon_from_bbox(annotation.bbox)


@dataclass
class AnnotationSessions:
    """Train/test annotation sessions for one episode."""

    episode: Episode
    assets: dict[str, object]
    train_session: "AnnotationSession"
    test_session: "AnnotationSession"


@dataclass
class BrowserAnnotationSessions:
    """Notebook-browser annotation state for one episode."""

    episode: Episode
    assets: dict[str, object]
    train_key: str
    test_key: str


class AnnotationSession:
    """
    Interactive polygon annotator for notebook use.

    Controls:
    - left click: add point
    - right click: undo last point
    - `enter` or `c`: close polygon
    - `backspace` / `delete`: undo
    - `r`: reset
    - `o`: reopen a closed polygon for editing
    """

    def __init__(
        self,
        image_bgr: np.ndarray,
        *,
        title: str,
        variant: str,
        view: str,
        object_name: str,
        initial_polygon: list[list[int]] | None = None,
    ) -> None:
        self.image_bgr = image_bgr.copy()
        self.title = title
        self.variant = variant
        self.view = view
        self.object_name = object_name
        self.points: list[list[int]] = [list(map(int, pt)) for pt in (initial_polygon or [])]
        self.closed = len(self.points) >= 3
        self.fig = None
        self.ax = None
        self._cid_click = None
        self._cid_key = None

    def show(self, figsize: tuple[int, int] = (9, 6)) -> "AnnotationSession":
        """Open an interactive figure."""
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._cid_click = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._redraw()
        plt.show()
        return self

    def _on_click(self, event) -> None:
        if self.ax is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            if self.closed:
                return
            self.points.append([int(round(event.xdata)), int(round(event.ydata))])
        elif event.button == 3:
            if self.points:
                self.points.pop()
            self.closed = False
        self._redraw()

    def _on_key(self, event) -> None:
        if event.key in {"enter", "c"}:
            if len(self.points) >= 3:
                self.closed = True
        elif event.key in {"backspace", "delete"}:
            if self.points:
                self.points.pop()
            self.closed = False
        elif event.key == "r":
            self.points = []
            self.closed = False
        elif event.key == "o":
            self.closed = False
        self._redraw()

    def _render_canvas(self) -> np.ndarray:
        canvas = self.image_bgr.copy()
        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(canvas, [pts], self.closed, (0, 255, 255), 2)
        for idx, (x, y) in enumerate(self.points):
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                str(idx + 1),
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if self.closed and len(self.points) >= 3:
            mask = self.get_mask()
            overlay = canvas.copy()
            overlay[mask > 0] = (0, 180, 0)
            canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.9, 0.0)
            x0, y0, x1, y1 = self.get_bbox()
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 0, 255), 3)
        return canvas

    def _redraw(self) -> None:
        if self.ax is None:
            return
        self.ax.clear()
        canvas = self._render_canvas()
        self.ax.imshow(_display_rgb(canvas))
        state = "closed" if self.closed else "open"
        subtitle = (
            "Left click: add point | Right click: undo | Enter/C: close | O: reopen | R: reset"
        )
        self.ax.set_title(f"{self.title} ({state})\n{subtitle}")
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def get_polygon(self) -> list[list[int]]:
        """Return the current polygon."""
        return [list(map(int, pt)) for pt in self.points]

    def get_mask(self) -> np.ndarray:
        """Return a binary mask for the current polygon."""
        if len(self.points) < 3:
            raise ValueError(f"{self.title}: need at least 3 points before saving.")
        return mask_from_polygon(self.image_bgr.shape[:2], self.get_polygon())

    def get_bbox(self) -> list[int]:
        """Return the bbox derived from the current polygon mask."""
        return bbox_from_mask(self.get_mask())

    def save_annotation(self, mask_output_path: str | Path) -> dict[str, object]:
        """Persist the current polygon as a mask-backed annotation payload."""
        polygon = self.get_polygon()
        mask_path = Path(mask_output_path)
        if not mask_path.is_absolute():
            mask_path = PROJECT_ROOT / mask_path
        mask = self.get_mask()
        save_mask(mask, mask_path)
        return annotation_to_serializable(
            variant=self.variant,
            view=self.view,
            mask_path=str(mask_path),
            bbox=bbox_from_mask(mask),
            polygon=polygon,
        )


def set_notebook_polygon(key: str, polygon: list[list[int]]) -> None:
    """Store a polygon from browser-side JS into Python memory."""
    NOTEBOOK_POLYGONS[key] = [[int(x), int(y)] for x, y in polygon]


def set_notebook_polygon_from_json(key: str, polygon_json: str) -> list[list[int]]:
    """Store a polygon from a JSON string copied out of the notebook textarea."""
    polygon = json.loads(polygon_json)
    if not isinstance(polygon, list):
        raise ValueError("Polygon JSON must decode to a list of [x, y] points.")
    normalized: list[list[int]] = []
    for point in polygon:
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            raise ValueError("Each polygon point must be a two-item list like [x, y].")
        normalized.append([int(point[0]), int(point[1])])
    set_notebook_polygon(key, normalized)
    return normalized


def get_notebook_polygon(key: str) -> list[list[int]]:
    """Read a polygon stored by the browser annotator."""
    return [list(map(int, pt)) for pt in NOTEBOOK_POLYGONS.get(key, [])]


def clear_notebook_polygon(key: str) -> None:
    """Clear a stored notebook polygon."""
    NOTEBOOK_POLYGONS.pop(key, None)


def _normalize_bbox(bbox: list[int]) -> list[int]:
    x0, y0, x1, y1 = map(int, bbox)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def set_notebook_box(key: str, bbox: list[int]) -> None:
    """Store a bbox from browser-side JS into Python memory."""
    NOTEBOOK_BOXES[key] = _normalize_bbox(bbox)


def set_notebook_box_from_json(key: str, bbox_json: str) -> list[int]:
    """Store a bbox from a JSON string copied out of the notebook textarea."""
    bbox = json.loads(bbox_json)
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise ValueError("BBox JSON must decode to [x0, y0, x1, y1].")
    normalized = _normalize_bbox([int(v) for v in bbox])
    set_notebook_box(key, normalized)
    return normalized


def get_notebook_box(key: str) -> list[int]:
    """Read a bbox stored by the browser annotator."""
    return list(NOTEBOOK_BOXES.get(key, []))


def clear_notebook_box(key: str) -> None:
    """Clear a stored notebook bbox."""
    NOTEBOOK_BOXES.pop(key, None)


def _image_to_base64_png(image_bgr: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("Failed to encode image for notebook display.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _browser_annotator_html(
    *,
    key: str,
    title: str,
    image_bgr: np.ndarray,
    initial_polygon: list[list[int]] | None = None,
) -> str:
    image_b64 = _image_to_base64_png(image_bgr)
    points_json = json.dumps(initial_polygon or [])
    width = int(image_bgr.shape[1])
    height = int(image_bgr.shape[0])
    title_json = json.dumps(title)
    key_json = json.dumps(key)
    dom_key = "".join(ch if ch.isalnum() else "_" for ch in key)
    return f"""
<div style="margin: 12px 0 28px 0;">
  <div style="font-weight: 600; margin-bottom: 8px;">{title}</div>
  <canvas id="canvas_{dom_key}" width="{width}" height="{height}" style="border:1px solid #999; max-width:100%; cursor:crosshair;"></canvas>
  <div style="margin-top: 8px;">
    <button type="button" id="close_{dom_key}">Close</button>
    <button type="button" id="undo_{dom_key}">Undo</button>
    <button type="button" id="reopen_{dom_key}">Reopen</button>
    <button type="button" id="reset_{dom_key}">Reset</button>
  </div>
  <div id="status_{dom_key}" style="margin-top: 6px; font-size: 12px;">
    Waiting for points.
  </div>
  <div style="margin-top: 4px; font-size: 12px;">
    Left click: add point | Right click: undo | Double click or Close: finalize polygon
  </div>
  <textarea id="textarea_{dom_key}" style="width:100%; height:90px; margin-top:8px; font-family:monospace;"></textarea>
</div>
<script>
(function() {{
  const key = {key_json};
  const title = {title_json};
  const imageSrc = "data:image/png;base64,{image_b64}";
  const initialPoints = {points_json};
  window.rceAnnotators = window.rceAnnotators || {{}};
  const canvas = document.getElementById("canvas_{dom_key}");
  const textarea = document.getElementById("textarea_{dom_key}");
  const status = document.getElementById("status_{dom_key}");
  const closeButton = document.getElementById("close_{dom_key}");
  const undoButton = document.getElementById("undo_{dom_key}");
  const reopenButton = document.getElementById("reopen_{dom_key}");
  const resetButton = document.getElementById("reset_{dom_key}");
  const ctx = canvas.getContext("2d");
  const image = new Image();
  const state = {{
    points: initialPoints.map((pt) => [Math.round(pt[0]), Math.round(pt[1])]),
    closed: initialPoints.length >= 3,
  }};

  function kernelPush() {{
    textarea.value = JSON.stringify(state.points);
    if (window.Jupyter && Jupyter.notebook && Jupyter.notebook.kernel) {{
      const cmd = "from rce_benchmark.notebook_tools import set_notebook_polygon\\n" +
        "set_notebook_polygon(" + JSON.stringify(key) + ", " + JSON.stringify(state.points) + ")";
      Jupyter.notebook.kernel.execute(cmd);
    }}
  }}

  function updateStatus(extra) {{
    const mode = state.closed ? "closed" : "open";
    const message = extra ? " | " + extra : "";
    status.textContent = `${{title}}: ${{state.points.length}} points | ${{mode}}${{message}}`;
  }}

  function draw() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);
    if (state.closed && state.points.length >= 3) {{
      ctx.fillStyle = "rgba(30, 180, 60, 0.25)";
      ctx.beginPath();
      ctx.moveTo(state.points[0][0], state.points[0][1]);
      for (let i = 1; i < state.points.length; i += 1) {{
        ctx.lineTo(state.points[i][0], state.points[i][1]);
      }}
      ctx.closePath();
      ctx.fill();

      const xs = state.points.map((pt) => pt[0]);
      const ys = state.points.map((pt) => pt[1]);
      const x0 = Math.min(...xs);
      const y0 = Math.min(...ys);
      const x1 = Math.max(...xs);
      const y1 = Math.max(...ys);
      ctx.strokeStyle = "rgba(255, 0, 255, 0.95)";
      ctx.lineWidth = 3;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }}

    if (state.points.length >= 2) {{
      ctx.strokeStyle = "rgba(255, 220, 0, 0.95)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(state.points[0][0], state.points[0][1]);
      for (let i = 1; i < state.points.length; i += 1) {{
        ctx.lineTo(state.points[i][0], state.points[i][1]);
      }}
      if (state.closed && state.points.length >= 3) {{
        ctx.closePath();
      }}
      ctx.stroke();
    }}

    for (let i = 0; i < state.points.length; i += 1) {{
      const [x, y] = state.points[i];
      ctx.fillStyle = "rgba(255, 220, 0, 1.0)";
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = "white";
      ctx.font = "12px sans-serif";
      ctx.fillText(String(i + 1), x + 7, y - 7);
    }}
    kernelPush();
    updateStatus();
  }}

  function clientPoint(evt) {{
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return [
      Math.round((evt.clientX - rect.left) * scaleX),
      Math.round((evt.clientY - rect.top) * scaleY),
    ];
  }}

  canvas.addEventListener("click", (evt) => {{
    if (state.closed) return;
    state.points.push(clientPoint(evt));
    draw();
  }});
  canvas.addEventListener("dblclick", (evt) => {{
    evt.preventDefault();
    if (state.points.length >= 3) {{
      state.closed = true;
      draw();
      updateStatus("double-click finalized");
    }}
  }});
  canvas.addEventListener("contextmenu", (evt) => {{
    evt.preventDefault();
    if (state.points.length > 0) {{
      state.points.pop();
      state.closed = false;
      draw();
    }}
  }});

  closeButton.addEventListener("click", () => {{
    if (state.points.length >= 3) {{
      state.closed = true;
      draw();
      updateStatus("closed by button");
    }} else {{
      updateStatus("need at least 3 points");
    }}
  }});
  undoButton.addEventListener("click", () => {{
    if (state.points.length > 0) {{
      state.points.pop();
      state.closed = false;
      draw();
      updateStatus("undid last point");
    }}
  }});
  reopenButton.addEventListener("click", () => {{
    state.closed = false;
    draw();
    updateStatus("reopened for editing");
  }});
  resetButton.addEventListener("click", () => {{
    state.points = [];
    state.closed = false;
    draw();
    updateStatus("reset polygon");
  }});

  window.rceAnnotators[key] = {{
    closePolygon: () => {{
      if (state.points.length >= 3) {{
        state.closed = true;
        draw();
        updateStatus("closed programmatically");
      }}
    }},
    undo: () => {{
      if (state.points.length > 0) {{
        state.points.pop();
        state.closed = false;
        draw();
        updateStatus("undid last point");
      }}
    }},
    reopen: () => {{
      state.closed = false;
      draw();
      updateStatus("reopened for editing");
    }},
    reset: () => {{
      state.points = [];
      state.closed = false;
      draw();
      updateStatus("reset polygon");
    }},
  }};

  image.onload = draw;
  image.src = imageSrc;
}})();
</script>
"""


def _browser_box_annotator_html(
    *,
    key: str,
    title: str,
    image_bgr: np.ndarray,
    initial_bbox: list[int] | None = None,
) -> str:
    image_b64 = _image_to_base64_png(image_bgr)
    bbox_json = json.dumps(initial_bbox or [])
    width = int(image_bgr.shape[1])
    height = int(image_bgr.shape[0])
    title_json = json.dumps(title)
    key_json = json.dumps(key)
    dom_key = "".join(ch if ch.isalnum() else "_" for ch in key)
    return f"""
<div style="margin: 12px 0 28px 0;">
  <div style="font-weight: 600; margin-bottom: 8px;">{title}</div>
  <canvas id="box_canvas_{dom_key}" width="{width}" height="{height}" style="border:1px solid #999; max-width:100%; cursor:crosshair;"></canvas>
  <div style="margin-top: 8px;">
    <button type="button" id="box_reset_{dom_key}">Reset</button>
  </div>
  <div id="box_status_{dom_key}" style="margin-top: 6px; font-size: 12px;">
    Drag on the image to create a rectangle. Drag inside to move, drag handles to resize.
  </div>
  <div style="margin-top: 4px; font-size: 12px;">
    Source of truth is the rectangle: [x0, y0, x1, y1]
  </div>
  <textarea id="box_textarea_{dom_key}" style="width:100%; height:70px; margin-top:8px; font-family:monospace;"></textarea>
</div>
<script>
(function() {{
  const key = {key_json};
  const title = {title_json};
  const imageSrc = "data:image/png;base64,{image_b64}";
  const initialRect = {bbox_json};
  window.rceBoxAnnotators = window.rceBoxAnnotators || {{}};
  const canvas = document.getElementById("box_canvas_{dom_key}");
  const textarea = document.getElementById("box_textarea_{dom_key}");
  const status = document.getElementById("box_status_{dom_key}");
  const resetButton = document.getElementById("box_reset_{dom_key}");
  const ctx = canvas.getContext("2d");
  const image = new Image();
  const state = {{
    rect: initialRect.length === 4 ? initialRect.slice() : null,
    dragMode: null,
    dragStart: null,
    originalRect: null,
  }};
  const HANDLE = 10;

  function normalizeRect(rect) {{
    const x0 = Math.min(rect[0], rect[2]);
    const y0 = Math.min(rect[1], rect[3]);
    const x1 = Math.max(rect[0], rect[2]);
    const y1 = Math.max(rect[1], rect[3]);
    return [Math.round(x0), Math.round(y0), Math.round(x1), Math.round(y1)];
  }}

  function pushRect() {{
    textarea.value = state.rect ? JSON.stringify(normalizeRect(state.rect)) : "";
    if (state.rect && window.Jupyter && Jupyter.notebook && Jupyter.notebook.kernel) {{
      const cmd = "from rce_benchmark.notebook_tools import set_notebook_box\\n" +
        "set_notebook_box(" + JSON.stringify(key) + ", " + JSON.stringify(normalizeRect(state.rect)) + ")";
      Jupyter.notebook.kernel.execute(cmd);
    }}
  }}

  function updateStatus(extra) {{
    if (!state.rect) {{
      status.textContent = `${{title}}: no rectangle${{extra ? " | " + extra : ""}}`;
      return;
    }}
    const r = normalizeRect(state.rect);
    status.textContent = `${{title}}: [${{r.join(", ")}}]${{extra ? " | " + extra : ""}}`;
  }}

  function drawHandle(x, y) {{
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.strokeStyle = "rgba(255,0,255,0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.rect(x - HANDLE / 2, y - HANDLE / 2, HANDLE, HANDLE);
    ctx.fill();
    ctx.stroke();
  }}

  function draw() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);
    if (state.rect) {{
      const [x0, y0, x1, y1] = normalizeRect(state.rect);
      ctx.fillStyle = "rgba(30, 180, 60, 0.18)";
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeStyle = "rgba(255, 0, 255, 0.95)";
      ctx.lineWidth = 3;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].forEach(([x, y]) => drawHandle(x, y));
    }}
    pushRect();
    updateStatus();
  }}

  function clientPoint(evt) {{
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return [
      Math.round((evt.clientX - rect.left) * scaleX),
      Math.round((evt.clientY - rect.top) * scaleY),
    ];
  }}

  function hitMode(pt) {{
    if (!state.rect) return "create";
    const [x0, y0, x1, y1] = normalizeRect(state.rect);
    const handles = {{
      nw: [x0, y0],
      ne: [x1, y0],
      se: [x1, y1],
      sw: [x0, y1],
    }};
    for (const [name, handle] of Object.entries(handles)) {{
      if (Math.abs(pt[0] - handle[0]) <= HANDLE && Math.abs(pt[1] - handle[1]) <= HANDLE) {{
        return name;
      }}
    }}
    if (pt[0] >= x0 && pt[0] <= x1 && pt[1] >= y0 && pt[1] <= y1) {{
      return "move";
    }}
    return "create";
  }}

  canvas.addEventListener("mousedown", (evt) => {{
    evt.preventDefault();
    const pt = clientPoint(evt);
    state.dragMode = hitMode(pt);
    state.dragStart = pt;
    state.originalRect = state.rect ? normalizeRect(state.rect) : null;
    if (state.dragMode === "create") {{
      state.rect = [pt[0], pt[1], pt[0], pt[1]];
      state.originalRect = state.rect.slice();
      draw();
    }}
  }});

  canvas.addEventListener("mousemove", (evt) => {{
    if (!state.dragMode || !state.rect || !state.dragStart) return;
    const pt = clientPoint(evt);
    const base = state.originalRect ? state.originalRect.slice() : state.rect.slice();
    if (state.dragMode === "create") {{
      state.rect = [base[0], base[1], pt[0], pt[1]];
    }} else if (state.dragMode === "move") {{
      const dx = pt[0] - state.dragStart[0];
      const dy = pt[1] - state.dragStart[1];
      state.rect = [base[0] + dx, base[1] + dy, base[2] + dx, base[3] + dy];
    }} else if (state.dragMode === "nw") {{
      state.rect = [pt[0], pt[1], base[2], base[3]];
    }} else if (state.dragMode === "ne") {{
      state.rect = [base[0], pt[1], pt[0], base[3]];
    }} else if (state.dragMode === "se") {{
      state.rect = [base[0], base[1], pt[0], pt[1]];
    }} else if (state.dragMode === "sw") {{
      state.rect = [pt[0], base[1], base[2], pt[1]];
    }}
    draw();
  }});

  function endDrag(extra) {{
    if (state.rect) {{
      state.rect = normalizeRect(state.rect);
      draw();
      updateStatus(extra);
    }}
    state.dragMode = null;
    state.dragStart = null;
    state.originalRect = null;
  }}

  canvas.addEventListener("mouseup", () => endDrag("saved to textarea"));
  canvas.addEventListener("mouseleave", () => {{
    if (state.dragMode) endDrag("saved to textarea");
  }});
  canvas.addEventListener("contextmenu", (evt) => evt.preventDefault());
  resetButton.addEventListener("click", () => {{
    state.rect = null;
    state.dragMode = null;
    state.dragStart = null;
    state.originalRect = null;
    textarea.value = "";
    draw();
    updateStatus("reset");
  }});

  window.rceBoxAnnotators[key] = {{
    getRect: () => state.rect ? normalizeRect(state.rect) : null,
    reset: () => resetButton.click(),
  }};

  image.onload = draw;
  image.src = imageSrc;
}})();
</script>
"""


def open_browser_annotation_sessions(
    manifest_path: str | Path,
    episode_id: str,
) -> BrowserAnnotationSessions:
    """Open browser-native polygon annotators inside notebook output cells."""
    from IPython.display import HTML, display

    episode = load_episode_by_id(manifest_path, episode_id)
    assets = load_episode_assets(episode)
    train_key = f"{episode_id}:train"
    test_key = f"{episode_id}:test"
    clear_notebook_polygon(train_key)
    clear_notebook_polygon(test_key)
    display(
        HTML(
            _browser_annotator_html(
                key=train_key,
                title=f"{episode.episode_id} train | {episode.object_name}",
                image_bgr=assets["train_image"],
                initial_polygon=_annotation_polygon(episode.train),
            )
        )
    )
    display(
        HTML(
            _browser_annotator_html(
                key=test_key,
                title=f"{episode.episode_id} test | {episode.object_name}",
                image_bgr=assets["test_image"],
                initial_polygon=_annotation_polygon(episode.test),
            )
        )
    )
    return BrowserAnnotationSessions(
        episode=episode,
        assets=assets,
        train_key=train_key,
        test_key=test_key,
    )


def open_browser_box_sessions(
    manifest_path: str | Path,
    episode_id: str,
) -> SimpleNamespace:
    """Open browser-native bbox annotators inside notebook output cells."""
    from IPython.display import HTML, display

    episode = load_episode_by_id(manifest_path, episode_id)
    assets = load_episode_assets(episode)
    train_key = f"{episode_id}:train_box"
    test_key = f"{episode_id}:test_box"
    clear_notebook_box(train_key)
    clear_notebook_box(test_key)
    display(
        HTML(
            _browser_box_annotator_html(
                key=train_key,
                title=f"{episode.episode_id} train box | {episode.object_name}",
                image_bgr=assets["train_image"],
                initial_bbox=episode.train_bbox,
            )
        )
    )
    display(
        HTML(
            _browser_box_annotator_html(
                key=test_key,
                title=f"{episode.episode_id} test box | {episode.object_name}",
                image_bgr=assets["test_image"],
                initial_bbox=episode.test_bbox,
            )
        )
    )
    return SimpleNamespace(
        episode=episode,
        assets=assets,
        train_key=train_key,
        test_key=test_key,
    )


def save_browser_annotation_sessions(
    sessions: BrowserAnnotationSessions,
    *,
    mask_dir: str | Path | None = None,
    mask_prefix: str | None = None,
    export_json_path: str | Path | None = None,
) -> dict[str, object]:
    """Save polygons drawn in browser annotators as mask-backed annotations."""
    episode = sessions.episode
    train_polygon = get_notebook_polygon(sessions.train_key)
    test_polygon = get_notebook_polygon(sessions.test_key)
    if len(train_polygon) < 3:
        raise ValueError("Train polygon has not been finalized yet. Close the train polygon first.")
    if len(test_polygon) < 3:
        raise ValueError("Test polygon has not been finalized yet. Close the test polygon first.")

    mask_root = Path(mask_dir or (PROJECT_ROOT / "rce_benchmark" / "annotations" / "masks"))
    prefix = mask_prefix or episode.episode_id
    train_mask_path = mask_root / f"{prefix}_train.png"
    test_mask_path = mask_root / f"{prefix}_test.png"
    train_annotation = polygon_to_annotation(
        variant=episode.train_variant,
        view=episode.train_view,
        polygon=train_polygon,
        image_shape=sessions.assets["train_image"].shape[:2],
        mask_output_path=train_mask_path,
    )
    test_annotation = polygon_to_annotation(
        variant=episode.test_variant,
        view=episode.test_view,
        polygon=test_polygon,
        image_shape=sessions.assets["test_image"].shape[:2],
        mask_output_path=test_mask_path,
    )

    payload = {
        "train_annotation": train_annotation,
        "test_annotation": test_annotation,
        "train_mask_path": str(train_mask_path),
        "test_mask_path": str(test_mask_path),
    }
    if export_json_path is not None:
        payload["export_json_path"] = str(
            export_episode_payload(
                export_json_path,
                episode_id=episode.episode_id,
                object_id=episode.object_id,
                object_name=episode.object_name,
                scene_group=episode.scene_group,
                stage=episode.stage,
                window_size=episode.window_size,
                train_variant=episode.train_variant,
                test_variant=episode.test_variant,
                train_view=episode.train_view,
                test_view=episode.test_view,
                train_annotation=train_annotation,
                test_annotation=test_annotation,
            )
        )
    return payload


def save_browser_box_sessions(
    sessions,
    *,
    export_json_path: str | Path | None = None,
) -> dict[str, object]:
    """Save browser-drawn rectangles as bbox-backed annotations."""
    episode = sessions.episode
    train_bbox = get_notebook_box(sessions.train_key)
    test_bbox = get_notebook_box(sessions.test_key)
    if len(train_bbox) != 4:
        raise ValueError("Train bbox has not been synced yet. Drag a rectangle or paste bbox JSON first.")
    if len(test_bbox) != 4:
        raise ValueError("Test bbox has not been synced yet. Drag a rectangle or paste bbox JSON first.")

    train_annotation = annotation_to_serializable(
        variant=episode.train_variant,
        view=episode.train_view,
        mask_path=None,
        bbox=_normalize_bbox(train_bbox),
    )
    test_annotation = annotation_to_serializable(
        variant=episode.test_variant,
        view=episode.test_view,
        mask_path=None,
        bbox=_normalize_bbox(test_bbox),
    )

    payload = {
        "train_annotation": train_annotation,
        "test_annotation": test_annotation,
        "train_bbox": train_annotation["bbox"],
        "test_bbox": test_annotation["bbox"],
    }
    if export_json_path is not None:
        payload["export_json_path"] = str(
            export_episode_payload(
                export_json_path,
                episode_id=episode.episode_id,
                object_id=episode.object_id,
                object_name=episode.object_name,
                scene_group=episode.scene_group,
                stage=episode.stage,
                window_size=episode.window_size,
                train_variant=episode.train_variant,
                test_variant=episode.test_variant,
                train_view=episode.train_view,
                test_view=episode.test_view,
                train_annotation=train_annotation,
                test_annotation=test_annotation,
            )
        )
    return payload


def load_episode_by_id(manifest_path: str | Path, episode_id: str):
    """Load one episode object by id."""
    episodes = load_episode_manifest(manifest_path)
    for episode in episodes:
        if episode.episode_id == episode_id:
            return episode
    raise KeyError(f"Episode not found: {episode_id}")


def preview_episode(manifest_path: str | Path, episode_id: str, figsize: tuple[int, int] = (14, 7)):
    """Display train/test images with their current masks."""
    episode = load_episode_by_id(manifest_path, episode_id)
    assets = load_episode_assets(episode)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, split_name, image_key, mask_key, bbox_key in [
        (axes[0], "Train", "train_image", "train_mask", "train_bbox"),
        (axes[1], "Test", "test_image", "test_mask", "test_bbox"),
    ]:
        image = assets[image_key].copy()
        mask = assets[mask_key]
        x0, y0, x1, y1 = assets[bbox_key]
        overlay = image.copy()
        overlay[mask > 0] = (0, 200, 0)
        image = cv2.addWeighted(overlay, 0.35, image, 0.85, 0.0)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 4)
        ax.imshow(_display_rgb(image))
        ax.set_title(f"{split_name}: {episode.object_name}")
        ax.axis("off")
    fig.tight_layout()
    return episode, assets


def open_annotation_sessions(
    manifest_path: str | Path,
    episode_id: str,
    *,
    figsize: tuple[int, int] = (9, 6),
) -> AnnotationSessions:
    """Open interactive train/test annotation sessions for one episode."""
    episode = load_episode_by_id(manifest_path, episode_id)
    assets = load_episode_assets(episode)
    train_session = AnnotationSession(
        assets["train_image"],
        title=f"{episode.episode_id} train",
        variant=episode.train_variant,
        view=episode.train_view,
        object_name=episode.object_name,
        initial_polygon=_annotation_polygon(episode.train),
    ).show(figsize=figsize)
    test_session = AnnotationSession(
        assets["test_image"],
        title=f"{episode.episode_id} test",
        variant=episode.test_variant,
        view=episode.test_view,
        object_name=episode.object_name,
        initial_polygon=_annotation_polygon(episode.test),
    ).show(figsize=figsize)
    return AnnotationSessions(
        episode=episode,
        assets=assets,
        train_session=train_session,
        test_session=test_session,
    )


def polygon_to_annotation(
    variant: str,
    view: str,
    polygon: list[list[int]],
    image_shape: tuple[int, int],
    mask_output_path: str | Path,
) -> dict[str, object]:
    """Turn a polygon into a saved mask annotation payload."""
    mask = mask_from_polygon(image_shape, polygon)
    save_mask(mask, mask_output_path)
    bbox = bbox_from_mask(mask)
    return annotation_to_serializable(
        variant=variant,
        view=view,
        mask_path=str(mask_output_path),
        bbox=bbox,
        polygon=polygon,
    )


def export_episode_payload(
    output_path: str | Path,
    *,
    episode_id: str,
    object_id: str,
    object_name: str,
    scene_group: str,
    stage: str,
    window_size: list[int],
    train_variant: str,
    test_variant: str,
    train_view: str,
    test_view: str,
    train_annotation: dict[str, object],
    test_annotation: dict[str, object],
) -> Path:
    """Write one benchmark-ready episode JSON file."""
    payload = {
        "episode_id": episode_id,
        "object_id": object_id,
        "object_name": object_name,
        "scene_group": scene_group,
        "stage": stage,
        "window_size": window_size,
        "train_variant": train_variant,
        "test_variant": test_variant,
        "train_view": train_view,
        "test_view": test_view,
        "train_annotation": train_annotation,
        "test_annotation": test_annotation,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([payload], indent=2), encoding="utf-8")
    return output


def update_manifest_episode_annotations(
    manifest_path: str | Path,
    episode_id: str,
    *,
    train_annotation: dict[str, object],
    test_annotation: dict[str, object],
    output_path: str | Path | None = None,
    inplace: bool = False,
) -> Path:
    """Write refined train/test annotations back to a manifest."""
    manifest = Path(manifest_path)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    updated = False
    for row in rows:
        if row["episode_id"] == episode_id:
            row["train_annotation"] = train_annotation
            row["test_annotation"] = test_annotation
            updated = True
            break
    if not updated:
        raise KeyError(f"Episode not found in manifest: {episode_id}")

    destination = manifest if inplace else Path(output_path or (manifest.parent / f"{manifest.stem}_refined.json"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return destination


def save_annotation_sessions(
    sessions: AnnotationSessions,
    *,
    mask_dir: str | Path | None = None,
    mask_prefix: str | None = None,
    export_json_path: str | Path | None = None,
) -> dict[str, object]:
    """Save both train/test masks and optionally export a one-episode manifest."""
    episode = sessions.episode
    mask_root = Path(mask_dir or (PROJECT_ROOT / "rce_benchmark" / "annotations" / "masks"))
    prefix = mask_prefix or episode.episode_id
    train_mask_path = mask_root / f"{prefix}_train.png"
    test_mask_path = mask_root / f"{prefix}_test.png"
    train_annotation = sessions.train_session.save_annotation(train_mask_path)
    test_annotation = sessions.test_session.save_annotation(test_mask_path)

    payload = {
        "train_annotation": train_annotation,
        "test_annotation": test_annotation,
        "train_mask_path": str(train_mask_path),
        "test_mask_path": str(test_mask_path),
    }
    if export_json_path is not None:
        payload["export_json_path"] = str(
            export_episode_payload(
                export_json_path,
                episode_id=episode.episode_id,
                object_id=episode.object_id,
                object_name=episode.object_name,
                scene_group=episode.scene_group,
                stage=episode.stage,
                window_size=episode.window_size,
                train_variant=episode.train_variant,
                test_variant=episode.test_variant,
                train_view=episode.train_view,
                test_view=episode.test_view,
                train_annotation=train_annotation,
                test_annotation=test_annotation,
            )
        )
    return payload


def run_notebook_episode(
    config_path: str | Path,
    episode_id: str,
    model_id: str = "rce_rgb",
    output_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
):
    """Run one episode through the benchmark from a notebook."""
    config = load_config(config_path)
    config_dir = Path(config["_config_path"]).resolve().parent.parent
    resolved_manifest = Path(manifest_path) if manifest_path is not None else Path(config["episodes"])
    if not resolved_manifest.is_absolute():
        resolved_manifest = config_dir / resolved_manifest
    episode = load_episode_by_id(resolved_manifest, episode_id)
    if output_dir is not None:
        config["output_dir"] = str(Path(output_dir))
    if config["stage"] == "stage1":
        rows = run_stage1_speed(config, [episode], [model_id])
    else:
        rows = run_stage2_middlebury(config, [episode], [model_id])
    return rows


def show_artifact(path: str | Path, figsize: tuple[int, int] = (10, 6)):
    """Display one saved artifact image inline."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    plt.figure(figsize=figsize)
    plt.imshow(_display_rgb(image))
    plt.axis("off")
    plt.tight_layout()


def show_result_bundle(row):
    """Display the key artifact bundle for one ResultRow."""
    for path in [
        row.overlay_path,
        row.heatmap_path,
        row.latent_plot_path,
        row.prototype_gallery_path,
    ]:
        if path:
            show_artifact(path)

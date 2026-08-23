#!/usr/bin/env python3
"""Build frontend/public/data/uk_subregions.json from OSM (Overpass).

OSM models the UK's constituent countries as admin_level=4 boundary
relations whose outer ways trace the coastlines. Instead of chaining the
ways into rings (brittle: shared-node graphs with gaps produce
self-intersecting "spaghetti" rings, which d3-geo renders as the
COMPLEMENT of the polygon — i.e. the whole globe), this script:

  1. rasterizes the outer-way nodes onto a coarse grid (land mask),
  2. flood-fills the exterior to get a solid land region,
  3. extracts the closed boundary polygon by walking directed boundary
     edges (each land/sea cell edge becomes one directed segment; the
     segments tile into perfect cycles — the result is always a simple,
     closed ring),
  4. simplifies with Douglas-Peucker and validates with the same d3-geo
     the frontend uses (small spherical area, single path subpath,
     bounded pixel extent).

The result is deliberately coarse: these polygons only need to look
right at world-map scale (~30 px wide), where a 0.1° grid is subpixel.

Emits one GeoJSON feature per constituent with a pseudo ISO-numeric id
(real 3166-1 numerics are 3 digits, so these can't collide):
  82601 England, 82602 Scotland, 82603 Northern Ireland
Wales has no WHO series in this project, so it is intentionally omitted
(the underlying GB base polygon covers it in neutral color).

Usage: make_uk_subregions.py <input-json | -> <output.json>
  (input "-" fetches from the Overpass API; a path reads a saved response)
"""
import json
import math
import os
import subprocess
import sys
import urllib.parse
import urllib.request

NAMES = {
    "England": "82601",
    "Scotland": "82602",
    "Northern Ireland": "82603",
}

# OSM relation ids (name tags are bilingual, e.g. "Alba / Scotland")
RELATION_IDS = {
    58447: "England",
    58446: "Scotland",
    156393: "Northern Ireland",
}

QUERY = """
[out:json][timeout:120];
area["name"="United Kingdom"]["admin_level"="2"]->.uk;
relation["boundary"="administrative"]["admin_level"="4"](area.uk);
out geom;
"""

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

GRID_STEP = 0.1  # degrees; subpixel at the map's scale


def fetch():
    payload = "data=" + urllib.parse.quote(QUERY)
    last_err = None
    for url in ENDPOINTS:
        try:
            req = urllib.request.Request(
                url,
                data=payload.encode(),
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "influenza-dashboard-map/1.0 (research project)",
                },
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  {url} failed: {e}", file=sys.stderr)
    raise SystemExit(f"All Overpass endpoints failed, last error: {last_err}")


def outer_way_points(rel):
    pts = []
    for m in rel["members"]:
        if m["type"] == "way" and m.get("role", "outer") == "outer" and "geometry" in m:
            for g in m["geometry"]:
                pts.append((g["lat"], g["lon"]))
    return pts


def solid_land_mask(pts):
    """Rasterize points onto a grid and fill the enclosed interior via a
    flood fill of the exterior."""
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    lat0, lat1 = min(lats), max(lats)
    lon0, lon1 = min(lons), max(lons)
    # pad one cell so the exterior flood fill has room to wrap around
    lat0 -= GRID_STEP
    lat1 += GRID_STEP
    lon0 -= GRID_STEP
    lon1 += GRID_STEP
    ny = int(math.ceil((lat1 - lat0) / GRID_STEP))
    nx = int(math.ceil((lon1 - lon0) / GRID_STEP))

    def cell(lat, lon):
        x = int((lon - lon0) / GRID_STEP)
        y = int((lat - lat0) / GRID_STEP)
        return x, y

    raw = [[0] * nx for _ in range(ny)]  # 1 = coast cell (contains a node)
    for lat, lon in pts:
        x, y = cell(lat, lon)
        raw[y][x] = 1

    # flood fill the exterior sea from every border cell
    exterior = [[False] * nx for _ in range(ny)]
    stack = []
    for x in range(nx):
        for y in (0, ny - 1):
            if not raw[y][x] and not exterior[y][x]:
                exterior[y][x] = True
                stack.append((x, y))
    for y in range(ny):
        for x in (0, nx - 1):
            if not raw[y][x] and not exterior[y][x]:
                exterior[y][x] = True
                stack.append((x, y))
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < nx and 0 <= ny_ < ny and not raw[ny_][nx_] and not exterior[ny_][nx_]:
                exterior[ny_][nx_] = True
                stack.append((nx_, ny_))

    land = [[(raw[y][x] == 1 or not exterior[y][x]) for x in range(nx)] for y in range(ny)]
    return land, nx, ny, (lon0, lat0)


def boundary_cycles(land, nx, ny, origin):
    """Jarnik's contour tracing: for each 8-connected land component, walk
    the outer boundary pixel-by-pixel (always stepping to the first land
    neighbor in clockwise order from the direction we came from). This
    always yields a closed, simple polygon of pixel centers — no
    self-intersection is possible, which is what keeps d3-geo from ever
    rendering the complement of the region (the whole globe).
    Returns a list of closed (lon, lat) rings, one per component."""
    lon0, lat0 = origin

    def is_land(x, y):
        return 0 <= x < nx and 0 <= y < ny and land[y][x]

    # 8-connected components
    comp = [[-1] * nx for _ in range(ny)]
    comps = []
    for y in range(ny):
        for x in range(nx):
            if not land[y][x] or comp[y][x] != -1:
                continue
            cid = len(comps)
            members = []
            stack = [(x, y)]
            comp[y][x] = cid
            while stack:
                cx, cy = stack.pop()
                members.append((cx, cy))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ncx, ncy = cx + dx, cy + dy
                        if is_land(ncx, ncy) and comp[ncy][ncx] == -1:
                            comp[ncy][ncx] = cid
                            stack.append((ncx, ncy))
            comps.append(members)

    # clockwise direction order starting from north
    CW = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))

    rings = []
    for members in comps:
        if len(members) < 8:  # skip specks (noisy 1-2 cell islands)
            continue
        landset = set(members)
        # start: topmost, then leftmost boundary pixel of the component
        top = max(cy for _, cy in members)
        start = min(cx for cx, cy in members if cy == top)
        cx, cy = start, top
        # initial "came from" direction: west
        prev_idx = 6  # index of W in CW
        ring = [(cx, cy)]
        for _ in range(len(landset) * 12 + 100):
            # first land neighbor clockwise from just after prev_idx
            nxt = None
            for k in range(8):
                idx = (prev_idx + 1 + k) % 8
                dx, dy = CW[idx]
                if (cx + dx, cy + dy) in landset:
                    nxt = (cx + dx, cy + dy, idx)
                    break
            if nxt is None:
                break  # should not happen for a solid component
            cx, cy = nxt[0], nxt[1]
            prev_idx = (nxt[2] + 4) % 8  # opposite direction
            ring.append((cx, cy))
            if (cx, cy) == (start, top):
                break
        if len(ring) >= 5 and ring[-1] == ring[0]:
            rings.append([(lon0 + x * GRID_STEP, lat0 + y * GRID_STEP) for x, y in ring])
    return rings


def ring_area(ring):
    """Shoelace area (absolute, in deg^2 — fine for comparison)."""
    a = 0.0
    for i in range(len(ring) - 1):
        a += ring[i][1] * ring[i + 1][0] - ring[i + 1][1] * ring[i][0]
    return abs(a) / 2.0


def douglas_peucker(pts, eps):
    if len(pts) < 3:
        return pts
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1e-12
    dmax, imax = 0.0, 0
    for i in range(1, len(pts) - 1):
        d = abs(dy * pts[i][0] - dx * pts[i][1] + x1 * y0 - y1 * x0) / length
        if d > dmax:
            dmax, imax = d, i
    if dmax > eps:
        left = douglas_peucker(pts[: imax + 1], eps)
        right = douglas_peucker(pts[imax:], eps)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def simplify(ring, target=220):
    # Douglas-Peucker on a closed ring (first == last) degenerates: the
    # first->last chord has zero length and every point reads as distance
    # 0, collapsing the ring to 2 points. Simplify the OPEN chain, then
    # re-close it.
    open_ring = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else list(ring)
    simplified = open_ring
    for eps in (0.03, 0.05, 0.08, 0.12, 0.2, 0.35):
        simplified = douglas_peucker(simplified, eps)
        if len(simplified) <= target:
            break
    if len(simplified) > target:
        n = max(2, len(simplified) // target)
        simplified = simplified[::n]
        if simplified[0] != open_ring[0]:
            simplified[0] = open_ring[0]
    simplified.append(simplified[0])  # close the ring
    return simplified


def to_feature(name, ring):
    # `ring` is a list of (lon, lat) — GeoJSON wants [lon, lat]
    coords = [[p[0], p[1]] for p in ring]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return {
        "id": NAMES[name],
        "type": "Feature",
        "properties": {"name": name},
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: make_uk_subregions.py <input-json | -> <output.json>")
    src = sys.argv[1]
    data = json.load(open(src)) if src != "-" else fetch()

    features = []
    for rel in data.get("elements", []):
        if rel.get("type") != "relation" or rel.get("id") not in RELATION_IDS:
            continue
        name = RELATION_IDS[rel["id"]]
        pts = outer_way_points(rel)
        if not pts:
            raise SystemExit(f"{name}: no outer way geometry")
        land, nx, ny, origin = solid_land_mask(pts)
        cycles = boundary_cycles(land, nx, ny, origin)
        if not cycles:
            raise SystemExit(f"{name}: no boundary cycle found")
        biggest = max(cycles, key=ring_area)
        ring = simplify(biggest)
        features.append(to_feature(name, ring))
        print(
            f"  {name}: {len(pts)} nodes, {len(cycles)} cycles, "
            f"biggest {len(biggest)} pts -> {len(ring)} pts"
        )

    if {f["id"] for f in features} != set(NAMES.values()):
        raise SystemExit(f"Expected 3 features, got: {[f['id'] for f in features]}")

    # Validate with the SAME d3-geo the frontend uses: each feature must
    # project to a single small subpath (a self-intersecting ring would
    # render as the complement of the polygon = the whole globe).
    validator = r"""
    const d3 = require('d3-geo');
    const fs = require('fs');
    const uk = JSON.parse(fs.readFileSync(process.argv[1]));
    const proj = d3.geoNaturalEarth1().translate([400, 210]).scale(160);
    const path = d3.geoPath(proj);
    let bad = 0;
    for (const f of uk.features) {
      const d = path(f) || '';
      const nums = d.match(/-?\d+\.?\d*/g) || [];
      const xs = [], ys = [];
      for (let i = 0; i < nums.length - 1; i += 2) { xs.push(+nums[i]); ys.push(+nums[i+1]); }
      const w = Math.max(...xs) - Math.min(...xs);
      const h = Math.max(...ys) - Math.min(...ys);
      const ms = (d.match(/M/g) || []).length;
      const area = d3.geoArea(f);
      const ok = ms === 1 && w < 150 && h < 150 && area < 0.01;
      if (!ok) bad++;
      console.log(`  ${f.id} ${f.properties.name}: area=${area.toFixed(5)} sr, Ms=${ms}, ${w.toFixed(0)}x${h.toFixed(0)} px ${ok ? 'OK' : 'REJECTED'}`);
    }
    if (bad) process.exit(1);
    """
    out = sys.argv[2]
    with open(out, "w") as tmp:
        json.dump({"type": "FeatureCollection", "features": features}, tmp, separators=(",", ":"))
    frontend_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend"))
    r = subprocess.run(["node", "-e", validator, os.path.abspath(out)], cwd=frontend_dir, capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"validation FAILED for {out} — not a safe map overlay")
    print(f"Wrote {out} (validated)")


if __name__ == "__main__":
    main()

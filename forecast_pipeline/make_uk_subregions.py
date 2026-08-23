#!/usr/bin/env python3
"""Build frontend/public/data/uk_subregions.json from OSM (Overpass).

OSM models the UK's constituent countries as admin_level=4 boundary relations.
We take each relation's outer ways, chain them into rings, keep the biggest
ring (drops holes), simplify with Douglas-Peucker, and emit one GeoJSON
feature per constituent with a pseudo ISO-numeric id (826xx — real ISO
3166-1 numerics are 3 digits, so these can't collide):

  82601 England, 82602 Scotland, 82603 Northern Ireland

Wales has no WHO series in this project, so it is intentionally omitted
(the underlying GB base polygon covers it in neutral color).
"""
import json
import math
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


def chain_rings(ways):
    """Chain ordered ways into closed rings (ways may need reversing)."""
    rings = []
    ring = []
    for way in ways:
        pts = list(way)
        if ring:
            # reverse way so it continues from the ring's end
            if pts[0] != ring[-1] and pts[-1] == ring[-1]:
                pts = list(reversed(pts))
            if pts[0] != ring[-1]:
                # disconnected (closed the current ring)
                if len(ring) >= 4:
                    rings.append(ring)
                ring = []
        ring.extend(pts[1:] if ring else pts)
    if len(ring) >= 4:
        rings.append(ring)
    return rings


def ring_area(ring):
    """Shoelace area (signed, in raw lat/lon units — fine for comparison)."""
    a = 0.0
    for i in range(len(ring) - 1):
        a += ring[i][1] * ring[i + 1][0] - ring[i + 1][1] * ring[i][0]
    return abs(a) / 2.0


def douglas_peucker(pts, eps):
    if len(pts) < 3:
        return pts
    # perpendicular distance of each point to the chord first->last
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


def main():
    # argv[1] = input ("-" to fetch from Overpass, or a path to a saved response)
    # argv[2] = output path
    if len(sys.argv) < 3:
        raise SystemExit("usage: make_uk_subregions.py <input|- > <output>")
    src = sys.argv[1]
    data = json.load(open(src)) if src != "-" else fetch()
    features = []
    for rel in data.get("elements", []):
        if rel.get("type") != "relation":
            continue
        name = rel.get("id") in RELATION_IDS and RELATION_IDS[rel["id"]] or rel["tags"].get("name")
        if name not in NAMES:
            continue
        ways = [
            [(g["lat"], g["lon"]) for g in m["geometry"]]
            for m in rel["members"]
            if m["type"] == "way" and m.get("role", "outer") == "outer" and "geometry" in m
        ]
        # relation members come in ring order
        rings = chain_rings(ways)
        if not rings:
            print(f"  {name}: no rings assembled, skipping", file=sys.stderr)
            continue
        biggest = max(rings, key=ring_area)
        # simplify until ~220 points or below (eps in degrees)
        simplified = biggest
        for eps in (0.01, 0.02, 0.04, 0.08, 0.15):
            simplified = douglas_peucker(simplified, eps)
            if len(simplified) <= 220:
                break
        # GeoJSON order: [lon, lat]; close the ring
        coords = [[p[1], p[0]] for p in simplified]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append(
            {
                "type": "Feature",
                "id": NAMES[name],
                "properties": {"name": name},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )
        print(f"  {name}: {len(rings)} rings, biggest {len(biggest)} pts -> {len(coords)} pts")

    if len(features) != 3:
        raise SystemExit(f"Expected 3 features, got {len(features)}: {[f['properties']['name'] for f in features]}")

    out = {"type": "FeatureCollection", "features": features}
    path = sys.argv[2]
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Wrote {path} ({sum(len(f['geometry']['coordinates'][0]) for f in features)} total pts)")


if __name__ == "__main__":
    main()

// The installed topojson-client is v3 (pulled in by react-simple-maps) and
// ships without type declarations. Minimal declaration for the single API
// we use (Topology/GeometryCollection -> GeoJSON FeatureCollection).
declare module "topojson-client" {
  export function feature(topology: unknown, object: unknown): any;
}

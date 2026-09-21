import os
import json
import pyproj
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import transform
from shapely.validation import explain_validity

# Import existing True 3D modules from your pipeline
from db_engine import CadastreDatabaseEngine, CADASTRE_SRID
from main import robust_solid_to_wkt

def project_to_utm(geom, source_epsg=4326, target_epsg=CADASTRE_SRID):
    """
    Transforms a Shapely geometry from standard GPS Lat/Lon (4326) 
    to the target UTM projection (e.g., 32610 for meters) used by the Cadastre.
    """
    project = pyproj.Transformer.from_crs(
        f"EPSG:{source_epsg}", 
        f"EPSG:{target_epsg}", 
        always_xy=True
    ).transform
    return transform(project, geom)

def process_and_register_polygon(db, poly, unit_id, solid):
    """
    Registers an independently measured/validated 3D B-Rep `solid` via the
    database engine.

    `poly` is the projected 2D GIS polygon and is REFERENCE / SEMANTIC /
    VALIDATION evidence only. It never creates, positions, trims, or
    determines any Z geometry; all 3D geometry comes from `solid`.
    """
    try:
        # Convert to OGC PolyhedralSurface Z
        wkt_string = robust_solid_to_wkt(solid, global_mirror=False)
        
        if wkt_string and "()" not in wkt_string:
            # Register into the Ledger
            ulpin = db.register_property(
                unit_id=unit_id,
                ogc_3d_wkt=wkt_string,
                tier_type="SURFACE",
                floor_level=0
            )
            return ulpin
            
    except Exception as e:
        print(f"⚠️ Failed to extrude/register parcel {unit_id}: {e}")
        
    return None

def ingest_municipal_geojson(geojson_path, solids_by_id=None):
    """
    Main ingestion routine: projects 2D GIS parcel boundaries and their IDs and
    registers them against independently measured 3D B-Rep solids.

    `solids_by_id` maps each parcel/part ID to its measured, validated solid.
    A GIS polygon with no supplied solid is reference-only and is NOT registered.
    """
    solids_by_id = solids_by_id or {}
    print("==================================================")
    print("🗺️  STARTING 2D GIS TO 3D CADASTRE INGESTION")
    print("==================================================")
    
    if not os.path.exists(geojson_path):
        print(f"❌ GeoJSON file not found at {geojson_path}")
        return

    print(f"📂 Loading GIS boundaries from {geojson_path}...")
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    features = data.get("features", [])
    if not features:
        print("⚠️ No features found in GeoJSON.")
        return

    print(f"🔍 Found {len(features)} land parcels. Projecting and validating...")
    
    registered_count = 0
    
    with CadastreDatabaseEngine() as db:
        for idx, feature in enumerate(features, start=1):
            props = feature.get("properties", {})
            # Use the surveyor's plot ID if it exists, otherwise generate one
            plot_id = props.get("plot_id", props.get("id", f"MUNICIPAL_PARCEL_{idx}"))
            
            raw_geom = shape(feature["geometry"])
            
            # Project from Lat/Lon to UTM Meters
            utm_geom = project_to_utm(raw_geom)

            # Invalid GIS geometry is rejected, never repaired.
            if not utm_geom.is_valid:
                print(f"⚠️ Rejecting {plot_id}: invalid GIS geometry ({explain_validity(utm_geom)})")
                continue
            
            # GeoJSON can contain Polygons or MultiPolygons
            polygons = []
            if isinstance(utm_geom, Polygon):
                polygons.append(utm_geom)
            elif isinstance(utm_geom, MultiPolygon):
                polygons.extend(list(utm_geom.geoms))
            else:
                print(f"⚠️ Skipping {plot_id}: Unsupported geometry type {utm_geom.geom_type}")
                continue
                
            for poly_idx, poly in enumerate(polygons):
                sub_id = plot_id if len(polygons) == 1 else f"{plot_id}_Part{poly_idx+1}"
                
                if poly.is_empty:
                    continue

                solid = solids_by_id.get(sub_id)
                if solid is None:
                    print(f"⚠️ {sub_id}: no measured 3D B-Rep supplied — GIS polygon is "
                          f"reference only, not registered.")
                    continue
                    
                print(f"   🏗️ Processing {sub_id}...")
                ulpin = process_and_register_polygon(db, poly, sub_id, solid)
                
                if ulpin:
                    registered_count += 1

    print("==================================================")
    print(f"🎉 Successfully ingested and minted {registered_count} 3D GIS Parcels!")
    print("==================================================")

if __name__ == "__main__":
    # Example usage: Ensure you have a 'parcels.geojson' in your working directory
    # containing standard EPSG:4326 Lat/Lon features.
    test_file = "data/gis_layers/municipal_parcels.geojson"
    
    # Create dummy file for testing if it doesn't exist
    if not os.path.exists(test_file):
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        dummy_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"plot_id": "PLOT_8842_A"},
                    "geometry": {
                        "type": "Polygon",
                        # Dummy coordinates mimicking Lat/Lon near San Francisco
                        "coordinates": [[[-122.4194, 37.7749], [-122.4190, 37.7749], [-122.4190, 37.7745], [-122.4194, 37.7745], [-122.4194, 37.7749]]]
                    }
                }
            ]
        }
        with open(test_file, 'w') as f:
            json.dump(dummy_geojson, f)
        print(f"ℹ️ Created sample GeoJSON at {test_file}")

    ingest_municipal_geojson(test_file)
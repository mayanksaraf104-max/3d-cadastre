# True 3D Vertical Cadastre & ULPIN

A true 3D cadastral pipeline for converting architectural and geospatial
evidence into georeferenced 3D cadastral entities.

## Core Pipeline

Input
→ AI blueprint vectorization
→ blueprint metrology / OCR
→ jurisdiction resolution
→ measured XYZ / LiDAR evidence
→ true 3D B-Rep reconstruction
→ B-Rep validation
→ clash / registration
→ PostGIS cadastral ledger
→ WebGL / GLB export

## True 3D Principle

The system does not fabricate vertical geometry from assumed floor
heights or storey counts.

3D reconstruction requires sufficient measured or explicitly supported
vertical evidence.

If the required evidence is unavailable, reconstruction is blocked
rather than generating a false 3D cadastral geometry.

## Requirements

- Python
- PostgreSQL/PostGIS
- OpenCASCADE
- GeoPandas
- LiDAR/XYZ data where required
- YOLO/AI model for blueprint vectorization

## Configuration

Copy:

`.env.example`

to:

`.env`

and provide the required local configuration.

## Data

Large GIS datasets, LiDAR datasets, training data and local model
artifacts are intentionally excluded from this repository.

## Status

The repository contains the current true-3D pipeline implementation.
Some inputs require external geospatial/LiDAR/model data that is not
distributed with the repository.
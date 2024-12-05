#include "CustomMagneticField.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>

CustomMagneticField::CustomMagneticField(const std::vector<G4ThreeVector>& points, const std::vector<G4ThreeVector>& fields, InterpolationType interpType)
    : fPoints(points), fFields(fields), fInterpType(interpType) {
    // Initialize grid parameters
    initializeGrid();
}

CustomMagneticField::~CustomMagneticField() {
}

void CustomMagneticField::initializeGrid() {
    // Calculate the grid size
    G4ThreeVector minCorner(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    G4ThreeVector maxCorner(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());

    for (const auto& point : fPoints) {
        if (point.x() < minCorner.x()) minCorner.setX(point.x());
        if (point.y() < minCorner.y()) minCorner.setY(point.y());
        if (point.z() < minCorner.z()) minCorner.setZ(point.z());
        if (point.x() > maxCorner.x()) maxCorner.setX(point.x());
        if (point.y() > maxCorner.y()) maxCorner.setY(point.y());
        if (point.z() > maxCorner.z()) maxCorner.setZ(point.z());
    }

    x_min = minCorner.x();
    y_min = minCorner.y();
    z_min = minCorner.z();

    // Calculate the number of grid points in each dimension
    nx = static_cast<int>(std::round((maxCorner.x() - minCorner.x()) / (fPoints[1].x() - fPoints[0].x())));
    ny = static_cast<int>(std::round((maxCorner.y() - minCorner.y()) / (fPoints[1].y() - fPoints[0].y())));
    nz = static_cast<int>(std::round((maxCorner.z() - minCorner.z()) / (fPoints[1].z() - fPoints[0].z())));

    dx_inv = 1.0 / (fPoints[1].x() - fPoints[0].x());
    dy_inv = 1.0 / (fPoints[1].y() - fPoints[0].y());
    dz_inv = 1.0 / (fPoints[1].z() - fPoints[0].z());
}

void CustomMagneticField::GetFieldValueNearestNeighbor(const G4double Point[4], G4double *Bfield) const {
    // Calculate nearest indices using integer arithmetic
    int i = (int)round((Point[0] - x_min) * dx_inv);
    int j = (int)round((Point[1] - y_min) * dy_inv);
    int k = (int)round((Point[2] - z_min) * dz_inv);

    // Check bounds
    if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) {
        Bfield[0] = Bfield[1] = Bfield[2] = 0.0;
        return; // Out of bounds
    }

    // Compute flat index
    int idx = i + nx * (j + ny * k);

    // Assign the nearest values
    Bfield[0] = fFields[idx].x();
    Bfield[1] = fFields[idx].y();
    Bfield[2] = fFields[idx].z();
}

void CustomMagneticField::GetFieldValue(const G4double Point[4], G4double *Bfield) const {
    if (fInterpType == NEAREST_NEIGHBOR) {
        GetFieldValueNearestNeighbor(Point, Bfield);
    } else {
        GetFieldValueLinear(Point, Bfield);
    }
}
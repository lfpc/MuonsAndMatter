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
    // Determine the quadrant of the point
    int quadrant = 0;
    if (Point[0] >= 0 && Point[1] >= 0) {
        quadrant = 1;
    } else if (Point[0] < 0 && Point[1] >= 0) {
        quadrant = 2;
    } else if (Point[0] < 0 && Point[1] < 0) {
        quadrant = 3;
    } else if (Point[0] >= 0 && Point[1] < 0) {
        quadrant = 4;
    }
    // Create a copy of Point and give it the value of the corresponding symmetry to the 1st quadrant
    G4double SymmetricPoint[4] = {Point[0], Point[1], Point[2], Point[3]};
    if (quadrant == 2) {
        SymmetricPoint[0] *= -1;
    } else if (quadrant == 3) {
        SymmetricPoint[0] *= -1;
        SymmetricPoint[1] *= -1;
    } else if (quadrant == 4) {
        SymmetricPoint[1] *= -1;
    }

    // Calculate nearest indices using integer arithmetic
    int i = (int)round((SymmetricPoint[0] - x_min) * dx_inv);
    int j = (int)round((SymmetricPoint[1] - y_min) * dy_inv);
    int k = (int)round((SymmetricPoint[2] - z_min) * dz_inv);

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

    // Apply symmetry to the magnetic field
    if (quadrant == 2 || quadrant == 4) {
        Bfield[0] = -Bfield[0];
    } 
}

void CustomMagneticField::GetFieldValueLinear(const G4double Point[4], G4double *Bfield) const {
    // TODO
}

void CustomMagneticField::GetFieldValue(const G4double Point[4], G4double *Bfield) const {
    if (fInterpType == NEAREST_NEIGHBOR) {
        GetFieldValueNearestNeighbor(Point, Bfield);
    } else {
        GetFieldValueLinear(Point, Bfield);
    }
}
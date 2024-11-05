#include "CustomMagneticField.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>

CustomMagneticField::CustomMagneticField(const std::vector<G4ThreeVector>& points, const std::vector<G4ThreeVector>& fields, InterpolationType interpType)
    : fPoints(points), fFields(fields), fInterpType(interpType) {}

CustomMagneticField::~CustomMagneticField() {}

void CustomMagneticField::GetFieldValue(const G4double Point[4], G4double *Bfield) const {
    if (fInterpType == NEAREST_NEIGHBOR) {
        GetFieldValueNearestNeighbor(Point, Bfield);
    } else if (fInterpType == LINEAR) {
        GetFieldValueLinear(Point, Bfield);
    }
}

void CustomMagneticField::GetFieldValueNearestNeighbor(const G4double Point[4], G4double *Bfield) const {
    // Initialize the magnetic field to zero
    Bfield[0] = 0.0;
    Bfield[1] = 0.0;
    Bfield[2] = 0.0;

    // Find the nearest point and use its magnetic field value
    double minDistance = std::numeric_limits<double>::max();
    size_t nearestIndex = 0;
    for (size_t i = 0; i < fPoints.size(); ++i) {
        double distance = (fPoints[i] - G4ThreeVector(Point[0], Point[1], Point[2])).mag();
        if (distance < minDistance) {
            minDistance = distance;
            nearestIndex = i;
            Bfield[0] = fFields[i].x();
            Bfield[1] = fFields[i].y();
            Bfield[2] = fFields[i].z();
        }
    }
    //std::cout << "Evaluated B field: (" << Bfield[0]/tesla << ", " << Bfield[1]/tesla << ", " << Bfield[2]/tesla << ")\n";
    //std::cout << "Position: (" << Point[0]/m << ", " << Point[1]/m << ", " << Point[2]/m << ")\n";
    //std::cout << "Nearest neighbor position: (" << fPoints[nearestIndex].x()/m << ", " << fPoints[nearestIndex].y()/m << ", " << fPoints[nearestIndex].z()/m << ")\n";
}

void CustomMagneticField::GetFieldValueLinear(const G4double Point[4], G4double *Bfield) const {
    // Initialize the magnetic field to zero
    Bfield[0] = 0.0;
    Bfield[1] = 0.0;
    Bfield[2] = 0.0;

    // Find the bounding box for interpolation
    G4ThreeVector p(Point[0], Point[1], Point[2]);
    G4ThreeVector pMin, pMax;
    G4ThreeVector B000, B001, B010, B011, B100, B101, B110, B111;

    bool foundBoundingBox = false;
    for (size_t i = 0; i < fPoints.size(); ++i) {
        if (fPoints[i].x() <= p.x() && fPoints[i].y() <= p.y() && fPoints[i].z() <= p.z()) {
            pMin = fPoints[i];
            B000 = fFields[i];
            foundBoundingBox = true;
        }
        if (fPoints[i].x() >= p.x() && fPoints[i].y() >= p.y() && fPoints[i].z() >= p.z()) {
            pMax = fPoints[i];
            B111 = fFields[i];
            break;
        }
    }

    if (!foundBoundingBox) {
        return; // Point is outside the defined field region
    }

    // Interpolate the magnetic field
    double xd = (p.x() - pMin.x()) / (pMax.x() - pMin.x());
    double yd = (p.y() - pMin.y()) / (pMax.y() - pMin.y());
    double zd = (p.z() - pMin.z()) / (pMax.z() - pMin.z());

    G4ThreeVector B00 = B000 * (1 - xd) + B100 * xd;
    G4ThreeVector B01 = B001 * (1 - xd) + B101 * xd;
    G4ThreeVector B10 = B010 * (1 - xd) + B110 * xd;
    G4ThreeVector B11 = B011 * (1 - xd) + B111 * xd;

    G4ThreeVector B0 = B00 * (1 - yd) + B10 * yd;
    G4ThreeVector B1 = B01 * (1 - yd) + B11 * yd;

    G4ThreeVector B = B0 * (1 - zd) + B1 * zd;

    Bfield[0] = B.x();
    Bfield[1] = B.y();
    Bfield[2] = B.z();

    std::cout << "Evaluated B field: (" << Bfield[0]/tesla << ", " << Bfield[1]/tesla << ", " << Bfield[2]/tesla << ")\n";
    std::cout << "Position: (" << Point[0] << ", " << Point[1] << ", " << Point[2] << ")\n";
    std::cout << "Bounding box min position: (" << pMin.x() << ", " << pMin.y() << ", " << pMin.z() << ")\n";
    std::cout << "Bounding box max position: (" << pMax.x() << ", " << pMax.y() << ", " << pMax.z() << ")\n";
    std::cout << "B field at min position: (" << B000.x() << ", " << B000.y() << ", " << B000.z() << ")\n";
    std::cout << "B field at max position: (" << B111.x() << ", " << B111.y() << ", " << B111.z() << ")\n";
}
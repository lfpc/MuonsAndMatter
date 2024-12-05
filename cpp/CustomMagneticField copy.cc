#include "CustomMagneticField.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>

CustomMagneticField::CustomMagneticField(const std::vector<G4ThreeVector>& points, const std::vector<G4ThreeVector>& fields, InterpolationType interpType)
    : fPoints(points), fFields(fields), fInterpType(interpType) {
    // Initialize bins
    initializeBins();
}

CustomMagneticField::~CustomMagneticField() {
}

void CustomMagneticField::initializeBins() {
    // Define the number of bins in each dimension
    numBinsX = 10;
    numBinsY = 10;
    numBinsZ = 10;

    // Calculate the bin size
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

    binSize = G4ThreeVector((maxCorner.x() - minCorner.x()) / numBinsX,
                            (maxCorner.y() - minCorner.y()) / numBinsY,
                            (maxCorner.z() - minCorner.z()) / numBinsZ);

    // Initialize bins
    bins.resize(numBinsX * numBinsY * numBinsZ);

    for (size_t i = 0; i < numBinsX; ++i) {
        for (size_t j = 0; j < numBinsY; ++j) {
            for (size_t k = 0; k < numBinsZ; ++k) {
                size_t binIndex = i + numBinsX * (j + numBinsY * k);
                bins[binIndex].minCorner = G4ThreeVector(minCorner.x() + i * binSize.x(),
                                                         minCorner.y() + j * binSize.y(),
                                                         minCorner.z() + k * binSize.z());
                bins[binIndex].maxCorner = G4ThreeVector(minCorner.x() + (i + 1) * binSize.x(),
                                                         minCorner.y() + (j + 1) * binSize.y(),
                                                         minCorner.z() + (k + 1) * binSize.z());

                // Find the nearest neighbors for this bin
                G4ThreeVector binCenter = (bins[binIndex].minCorner + bins[binIndex].maxCorner) / 2.0;
                std::vector<std::pair<double, size_t>> distances;
                for (size_t idx = 0; idx < fPoints.size(); ++idx) {
                    double dist = (fPoints[idx] - binCenter).mag2();
                    distances.emplace_back(dist, idx);
                }
                std::sort(distances.begin(), distances.end());
                for (size_t n = 0; n < 8 && n < distances.size(); ++n) {
                    bins[binIndex].nearestNeighbors.push_back(distances[n].second);
                }
            }
        }
    }
}

void CustomMagneticField::findBin(const G4ThreeVector& point, size_t& binX, size_t& binY, size_t& binZ) const {
    binX = std::min(static_cast<size_t>((point.x() - bins[0].minCorner.x()) / binSize.x()), numBinsX - 1);
    binY = std::min(static_cast<size_t>((point.y() - bins[0].minCorner.y()) / binSize.y()), numBinsY - 1);
    binZ = std::min(static_cast<size_t>((point.z() - bins[0].minCorner.z()) / binSize.z()), numBinsZ - 1);
}

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
    if (fFields.empty()) {
        // If fFields is empty, set the magnetic field to zero
        std::cout << "No magnetic field data available. Setting Bfield to zero." << std::endl;
        return;
    }

    G4ThreeVector point(Point[0], Point[1], Point[2]);

    size_t binX, binY, binZ;
    findBin(point, binX, binY, binZ);
    size_t binIndex = binX + numBinsX * (binY + numBinsY * binZ);

    const Bin& bin = bins[binIndex];
    double minDistance = std::numeric_limits<double>::max();
    size_t nearestNeighborIndex = 0;

    for (size_t index : bin.nearestNeighbors) {
        double distance = (point - fPoints[index]).mag2();
        if (distance < minDistance) {
            minDistance = distance;
            nearestNeighborIndex = index;
        }
    }

    Bfield[0] = fFields[nearestNeighborIndex].x();
    Bfield[1] = fFields[nearestNeighborIndex].y();
    Bfield[2] = fFields[nearestNeighborIndex].z();
    /*
    // Print the point where the inference is being made
    std::cout << "Inference Point: (" << Point[0]/m << ", " << Point[1]/m << ", " << Point[2]/m << ")" << std::endl;

    // Print the nearest point
    const G4ThreeVector& nearestPoint = fPoints[nearestNeighborIndex];
    std::cout << "Nearest Point: (" << nearestPoint.x()/m << ", " << nearestPoint.y()/m << ", " << nearestPoint.z()/m << ")" << std::endl;

    // Calculate and print the Euclidean distance between them
    double distance = (point - nearestPoint).mag();
    std::cout << "Euclidean Distance: " << distance/m << std::endl;

    // Print the Bfield used
    std::cout << "Bfield: (" << Bfield[0]/tesla << ", " << Bfield[1]/tesla << ", " << Bfield[2]/tesla << ")" << std::endl;*/
}

void CustomMagneticField::GetFieldValueLinear(const G4double Point[4], G4double *Bfield) const {
    // Initialize the magnetic field to zero
    Bfield[0] = 0.0;
    Bfield[1] = 0.0;
    Bfield[2] = 0.0;

    // Find the bin corresponding to the query point
    G4ThreeVector p(Point[0], Point[1], Point[2]);
    size_t binX, binY, binZ;
    findBin(p, binX, binY, binZ);
    size_t binIndex = binX + numBinsX * (binY + numBinsY * binZ);

    // Use the precomputed nearest neighbors for interpolation
    const auto& nearestNeighbors = bins[binIndex].nearestNeighbors;

    // Initialize bounding box corners
    G4ThreeVector pMin(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    G4ThreeVector pMax(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());

    // Find the bounding box corners
    for (const auto& index : nearestNeighbors) {
        const G4ThreeVector& pt = fPoints[index];
        if (pt.x() < pMin.x()) pMin.setX(pt.x());
        if (pt.y() < pMin.y()) pMin.setY(pt.y());
        if (pt.z() < pMin.z()) pMin.setZ(pt.z());
        if (pt.x() > pMax.x()) pMax.setX(pt.x());
        if (pt.y() > pMax.y()) pMax.setY(pt.y());
        if (pt.z() > pMax.z()) pMax.setZ(pt.z());
    }

    // Interpolate the magnetic field
    double xd = (Point[0] - pMin.x()) / (pMax.x() - pMin.x());
    double yd = (Point[1] - pMin.y()) / (pMax.y() - pMin.y());
    double zd = (Point[2] - pMin.z()) / (pMax.z() - pMin.z());

    G4ThreeVector B000 = fFields[nearestNeighbors[0]];
    G4ThreeVector B001 = fFields[nearestNeighbors[1]];
    G4ThreeVector B010 = fFields[nearestNeighbors[2]];
    G4ThreeVector B011 = fFields[nearestNeighbors[3]];
    G4ThreeVector B100 = fFields[nearestNeighbors[4]];
    G4ThreeVector B101 = fFields[nearestNeighbors[5]];
    G4ThreeVector B110 = fFields[nearestNeighbors[6]];
    G4ThreeVector B111 = fFields[nearestNeighbors[7]];

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
}